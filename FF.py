# FEEDFORWARD MODEL
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


# main hyperparameters
#---------------------------------------------------------------------------------------------------------------------------------#
FFW_Act    = "swish" # from Optuna
FFW_L2     = 6e-5
FFW_Layers = [256, 48, 208, 16]   # from Optuna (in order)
#---------------------------------------------------------------------------------------------------------------------------------#


################################################################
################################################################
def crear_modelo(input_shape):

    reg = regularizers.l2(FFW_L2)   # L2 regularization (manually optimized)

    model = Sequential()

    # input layer
    #---------------------------------------------------------------------------------------------------------------------------------#
    model.add(Input(shape=input_shape))
    #---------------------------------------------------------------------------------------------------------------------------------#

    model.add(Dense(FFW_Layers[0], activation=FFW_Act, kernel_regularizer=reg))
    #model.add(Dropout(0.05)) # does not improve even with small values

    model.add(Dense(FFW_Layers[1], activation=FFW_Act))

    model.add(Dense(FFW_Layers[2], activation=FFW_Act, kernel_regularizer=reg))

    model.add(Dense(FFW_Layers[3], activation=FFW_Act))

    #---------------------------------------------------------------------------------------------------------------------------------#
    model.add(Dense(1, activation="linear"))  # output - T prediction


    return model
################################################################
################################################################
