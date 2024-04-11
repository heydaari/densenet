import tensorflow as tf
from keras.layers import Dense, Dropout, BatchNormalization, ReLU, Conv2D, Concatenate


def densenet_block (x0):
    x1 = Conv2D(32, (3, 3), activation = 'relu', padding='same')(x0)
    x1 = BatchNormalization()(x1)

    x_x0 = Concatenate(axis=-1)([x, x0])
    x1 = Conv2D(32, (3, 3), activation = 'relu', padding='same')(x_x0)
    x1 = BatchNormalization()(x1)

    x_x0_x1 = Concatenate(axis=-1)([x, x0, x1])
    x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x_x0_x1)
    x2 = BatchNormalization()(x2)

    x_x0_x1_x2 = Concatenate(axis= -1)([x, x0, x1, x2])
    x3 = Conv2D(32, (3, 3), activation = 'relu', padding='same')(x_x0_x1_x2)
    x3 = BatchNormalization()(x3)

    x_x0_x1_x2_x3 = Concatenate(axis = -1)([x, x0, x1, x2, ])





