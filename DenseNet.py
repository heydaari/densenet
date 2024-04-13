
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization,
                                     Dense, AvgPool2D,
                                     ReLU, concatenate)

from keras.models import Model
import keras

# this function returns a block of batch-->relu-->conv for simplification
def batch_relu_conv(x, num_filters, conv_name, kernel_shape=(3, 3), strides=1):

    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(num_filters, kernel_shape, strides=strides, padding='same', name = conv_name)(x)
    return x

# this function returns a dense-conv block with k=num_convs conv layers
def dense_block(x, num_blocks, block_name):
    merge = []
    merge.append(x)
    for i in range(num_blocks):
        y = batch_relu_conv(x, num_filters= 64 , conv_name = f'{block_name}-dense_conv{i}')
        y = batch_relu_conv(y, num_filters= 64, conv_name =f'{block_name}-dense_relu{i}')
        merge.append(y)
        x = concatenate(merge)
    return x

# this function returns a transition block based on batch-relu-conv and avgpool
# which we use after every dense block
def transition_layer(x, name):

    from random import randint
    x = batch_relu_conv(x, num_filters= 128, conv_name = name)
    x = AvgPool2D(2, strides=2, padding='same')(x)
    return x


from keras.datasets import cifar10


def load_cifar10_data():

    num_classes = 10
    # Load cifar10 training and validation sets
    (X_train, Y_train), (X_valid, Y_valid) = cifar10.load_data()



    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')

    # preprocess data
    X_train = X_train / 255.0
    X_valid = X_valid / 255.0

    return X_train[:10000], Y_train[:10000], X_valid[:2000], Y_valid[:2000]

X_train, y_train, X_test, y_test = load_cifar10_data()
y_train.resize(y_train.shape[0])
y_test.resize(y_test.shape[0])

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Assuming X_train and X_test are your numpy arrays of images
# And each image in X_train and X_test is of shape (height, width, channels)

from tensorflow.image import resize

def resize_images(images):
    return resize(images, [128, 128])

X_train = resize_images(X_train)
X_test = resize_images(X_test)

print(X_train.shape, X_test.shape, sep = '\n')

# defining model with 3 dense blocks and k=4
x_input = Input(shape = (128, 128, 3))
x = transition_layer(x_input, name='transition_1')
x = dense_block(x, 4, block_name = 'block_1')
x = transition_layer(x, name= 'transition_2')
x = dense_block(x, 4, block_name = 'block_2')
x = transition_layer(x, name= 'transition_3')
x = dense_block(x, 4, block_name = 'block_3')
x = transition_layer(x, name= 'transition_4')
x = keras.layers.Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dense(10, activation='softmax')(x)

model = Model(x_input, x)
model.summary()

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 5, batch_size = 32)