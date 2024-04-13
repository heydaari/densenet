import tensorflow as tf
from keras.layers import Dense, Dropout, BatchNormalization, ReLU, Conv2D, Concatenate


def densenet_block (input_layer, num_convs):

    merge = []
    merge.append(input_layer)
    for i in range(num_convs):
        x = Conv2D(32, (3, 3), padding='same')(input_layer)
        x = ReLU()(x)
        x = BatchNormalization()(x)

        merge.append(x)
        x = Concatenate(axis= -1)(merge)

    return




