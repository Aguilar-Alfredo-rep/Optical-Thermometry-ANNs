# CONVOLUTIONAL MODEL

import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Layer

# main hyperparameters (from optuna)
#--------------------------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------#
Conv_Act = 'swish'
Dense_Act = 'swish'
Filters = [128, 128, 32]
Kernels = [7, 7, 7]
Pool = [2, 2, 2]

Att_Units = 128
Att_Heads = 1

Dense_Units = [56, 16, 8]

CNN_L2 = 6e-5  # manually adjusted
#--------------------------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------#


# fixed bands for attention (independent of λ_min, λ_max from main but in the same way)
#--------------------------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------#
B1_MIN, B1_MAX = 512.0, 536.0
B2_MIN, B2_MAX = 536.0, 556.0   # up to 556

# main sets this only once with cropped wavelengths
_W_CROP = None

def set_attention_wavelengths(w_crop_1d):
    global _W_CROP
    _W_CROP = np.asarray(w_crop_1d, dtype=np.float32).reshape(-1)
#--------------------------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------#


################################################################################
################################################################################
class SelfAttention(Layer):  #  band-token self-attention (builds 2 tokens: feature average in band1 and band2)
                             #  builds tok_rel (relative) and does cross-attention with softmax towards [tok_rel, tok_null] with residual gated

    def __init__(self, units, num_heads, **kwargs):
        super().__init__(**kwargs)
        assert units % num_heads == 0, "Att_Units must be divisible by Att_Heads"

        self.units = units
        self.num_heads = num_heads
        self.head_dim = units // num_heads

        # important: DO NOT put L2 here (avoids the confound you already detected)
        self.W_q = Dense(units, use_bias=False)
        self.W_k = Dense(units, use_bias=False)
        self.W_v = Dense(units, use_bias=False)
        self.out_proj = Dense(units, use_bias=False)

        self._mask1 = None
        self._mask2 = None
        self._last_weights = None
        self._token_w = None
        self.gamma = None
        self.att_tok_mean = None

    def build(self, input_shape):
        # input_shape: (B, T, C) at the point where attention is inserted (after conv2)
        T = int(input_shape[1])
        C = int(input_shape[-1])

        # to use residual "inputs + out", dimensions must match
        if C != self.units:
            raise ValueError(f"SelfAttention: units={self.units} must equal channels={C} to use residual inputs + out")

        # residual gate rezero / layerscale type (starts at 0 => does not perturb cnn at the beginning)
        self.gamma = self.add_weight(
            name="gamma_att",
            shape=(1, 1, self.units),
            initializer="zeros",
            trainable=True
        )

        # average attention weights towards [tok_rel, tok_null] (only for debug/print) - non-trainable
        self.att_tok_mean = self.add_weight(
            name="att_tok_mean",
            shape=(3,),
            initializer="zeros",
            trainable=False
        )

        # if wavelengths not set, silent fallback: half/half
        if _W_CROP is None or len(_W_CROP) < 10:
            m1 = np.zeros((T,), dtype=np.float32)
            m2 = np.zeros((T,), dtype=np.float32)
            split = T // 2
            m1[:split] = 1.0
            m2[split:] = 1.0
            self._mask1 = tf.constant(m1)
            self._mask2 = tf.constant(m2)
            super().build(input_shape)
            return

        w = _W_CROP
        L0 = len(w)

        # rough mapping conv2-token -> index in original cropped grid
        # assumes conv1d padding='valid' and maxpool stride=pool_size (keras default)
        K1 = int(Kernels[0])
        K2 = int(Kernels[1])
        P1 = int(Pool[0])

        k = np.arange(T, dtype=np.float32)
        center = (P1 * k) + (P1 * (K2 - 1) / 2.0) + ((P1 - 1) / 2.0) + ((K1 - 1) / 2.0)
        idx = np.clip(np.round(center).astype(int), 0, L0 - 1)
        token_w = w[idx]

        # bands in nm (half-open intervals)
        m1 = ((token_w >= B1_MIN) & (token_w <  B1_MAX)).astype(np.float32)
        m2 = ((token_w >= B2_MIN) & (token_w <  B2_MAX)).astype(np.float32)

        # fail-fast: if a band is empty due to λ_min/λ_max crop, warn explicitly
        if m1.sum() == 0 or m2.sum() == 0:
            raise ValueError(
                f"Empty bands in SelfAttention. "
                f"Available range in w_crop: [{float(w[0]):.3f}, {float(w[-1]):.3f}] nm. "
                f"B1=[{B1_MIN},{B1_MAX}), B2=[{B2_MIN},{B2_MAX})"
            )

        self._mask1 = tf.constant(m1, dtype=tf.float32)
        self._mask2 = tf.constant(m2, dtype=tf.float32)

        # (optional debug)
        self._token_w = token_w

        super().build(input_shape)

    def call(self, inputs):
        # inputs: (B, T, C)

        # tokens from inputs without ln (preserves feature amplitude/scale)
        x_tok = inputs

        B = tf.shape(inputs)[0]
        T = tf.shape(inputs)[1]

        m1 = tf.reshape(self._mask1, (1, T, 1))
        m2 = tf.reshape(self._mask2, (1, T, 1))
        eps = tf.constant(1e-6, dtype=inputs.dtype)

        # positive "intensity" proxy per channel (avoids signs and stabilizes log)
        x_pos = tf.nn.softplus(x_tok)  # >= 0, smooth, stable (better than abs() for gradients)

        # 2 band tokens: "area" (sum), NOT average
        tok1 = tf.reduce_sum(x_pos * m1, axis=1, keepdims=True)  # (B, 1, C)
        tok2 = tf.reduce_sum(x_pos * m2, axis=1, keepdims=True)  # (B, 1, C)

        # tok_rel (relative): log-ratio (analogous to ln(I1/I2))
        tok_rel = tf.math.log(tok1 + eps) - tf.math.log(tok2 + eps)   # (B, 1, C)

        # tok_abs (absolute): total energy (snr / "quality" proxy)
        tok_abs = tf.math.log(tok1 + tok2 + eps)  # (B, 1, C)

        # null token: baseline so attention can "ignore" thermometric info
        tok_null = tf.zeros_like(tok_rel)  # (B, 1, C)

        # cross-attention: sequence queries [tok_rel, tok_abs, tok_null]
        toks = tf.concat([tok_rel, tok_abs, tok_null], axis=1)  # (B, 3, C)

        Q = self.W_q(inputs)    # (B, T, C)
        K = self.W_k(toks)   # (B, 2, C)
        V = self.W_v(toks)   # (B, 2, C)

        def split_heads_q(z):
            z = tf.reshape(z, (B, T, self.num_heads, self.head_dim))
            return tf.transpose(z, perm=[0, 2, 1, 3])  # (B,H,T,D)

        def split_heads_kv(z):
            n_tok = tf.shape(z)[1]
            z = tf.reshape(z, (B, n_tok, self.num_heads, self.head_dim))
            return tf.transpose(z, perm=[0, 2, 1, 3])  # (B,H,n_tok,D)

        Qh = split_heads_q(Q)
        Kh = split_heads_kv(K)
        Vh = split_heads_kv(V)

        dk = tf.cast(self.head_dim, inputs.dtype)
        scores = tf.matmul(Qh, Kh, transpose_b=True) / tf.sqrt(dk)  # (B,H,T,3)
        weights = tf.nn.softmax(scores, axis=-1)

        # token attention average (averages batch, heads and sequence positions)
        att_tok_mean = tf.reduce_mean(weights, axis=[0, 1, 2])  # shape (2,)
        self.att_tok_mean.assign(tf.cast(att_tok_mean, self.att_tok_mean.dtype))

        att = tf.matmul(weights, Vh)  # (B,H,T,D)

        att = tf.transpose(att, perm=[0, 2, 1, 3])  # (B,T,H,D)
        att = tf.reshape(att, (B, T, self.units))

        out = self.out_proj(att)

        self._last_weights = weights  # shape (B,H,T,2)

        return inputs + self.gamma * out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'units': self.units, 'num_heads': self.num_heads})
        return cfg
################################################################################
################################################################################


################################################################################
################################################################################
def crear_modelo(input_shape, attention_units=Att_Units, num_heads=Att_Heads):

    reg = regularizers.l2(CNN_L2)

    model = Sequential()

    model.add(Input(shape=input_shape))

    # 1st conv layer
    model.add(Conv1D(filters=Filters[0], kernel_size=Kernels[0], activation=Conv_Act, padding='same'))
    model.add(MaxPooling1D(pool_size=Pool[0]))

    # 2nd conv layer
    model.add(Conv1D(filters=Filters[1], kernel_size=Kernels[1], activation=Conv_Act, padding='same'))
    #------------------------------------------------------------------------------------------------------------------------------------------#
    model.add(SelfAttention(units=attention_units, num_heads=num_heads))  # ------------------- attention module
    #------------------------------------------------------------------------------------------------------------------------------------------#
    model.add(MaxPooling1D(pool_size=Pool[1]))

    # 3rd conv layer
    model.add(Conv1D(filters=Filters[2], kernel_size=Kernels[2], activation=Conv_Act, padding='same'))
    model.add(MaxPooling1D(pool_size=Pool[2]))

    model.add(Flatten())

    # dense part (3 layers)
    model.add(Dense(Dense_Units[0], activation=Dense_Act, kernel_regularizer=reg))  # with L2 regularization
    model.add(Dense(Dense_Units[1], activation=Dense_Act))
    model.add(Dense(Dense_Units[2], activation=Dense_Act))

    # temperature output
    model.add(Dense(1, activation='linear'))

    return model
################################################################################
################################################################################
