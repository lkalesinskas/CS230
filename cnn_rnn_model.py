from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Conv3D, MaxPooling3D, LSTM, TimeDistributed
# from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from nilearn import plotting, image
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

K.set_image_data_format('channels_last')

data_path = 'EMC/'

files = [os.path.join(data_path, k) for k in os.listdir(data_path) if 'fmri' in k]
# y = np.random.randint(0, 2, size=len(files))

np.random.seed(10)
def batch_generator(images, n=1):
    for fmri in images:
        img = image.load_img(fmri).get_data()
        x, y, z, Tx = img.shape
        yield (img.reshape(Tx, 1, x, y, z), np.random.randint(low=0, high=1))


def make_model(input_shape):
    model = Sequential()
    Tx, x, y, z = input_shape

    model.add(TimeDistributed(Conv3D(filters=16,
                                     kernel_size=5,
                                     strides=1,
                                     padding='same'),
                              batch_input_shape=[85, 1, 64, 64, 31]))
    model.add(Activation('relu'))
    model.add(TimeDistributed(MaxPooling3D(pool_size=(5, 5, 5))))

    model.add(TimeDistributed(Conv3D(filters=32,
                                     kernel_size=3,
                                     strides=1,
                                     padding='same')))
    model.add(Activation('relu'))
    model.add(TimeDistributed(MaxPooling3D(pool_size=(5, 5, 5))))

    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(100, return_sequences=True))
    model.add(Dense(1, activation='sigmoid'))
    #probably add another time distributed convolution here: memory permitting

    return model



model = make_model((85, 64, 64, 31))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit_generator(generator=batch_generator(files),
                              steps_per_epoch=len(files))
plt.plot(history.history['acc'])
plt.show()



#
# X = Input((64, 64, 31))
#
# for time in range(Tx):
#
#     #LAYER 1 (convolution 5x5x5)
#     X = Conv3D(input_shape=[None, 1, 64, 64, 31],
#                      filters=16,
#                      kernel_size=5,
#                      strides=3,
#                      padding='same',
#                      kernel_initializer='glorot_uniform')(X)
#
#     X = Activation('relu')(X)
#
#     X = MaxPooling3D(pool_size=(5, 5, 5))(X)
#
#
#     #LAYER 2 (convolution 3x3x3)
#     X = Conv3D(filters=32,
#                      kernel_size=5,
#                      strides=3,
#                      padding='same',
#                      kernel_initializer='glorot_uniform')(X)
#
#     X = Activation('relu')(X)
#     X = MaxPooling3D(pool_size=(5, 5, 5))(X)
#
#
#     #LAYER 3(fully connected)
#     X = Flatten()(X)
#     X = Dense(model.output_shape[1],
#                     kernel_initializer='glorot_uniform',
#                     activation='relu')(X)
#     #LAYER 4(output)
#     X = Dense(32,
#                     kernel_initializer='uniform',
#                     activation='sigmoid')(X)
#
#     X = LSTM(32, return_sequence = True)(X)
#
#     X = LSTM(128, retrun_sequence = False)(X)
#
#     X = Dense(1)(X)
#
#     X = Activation ('sigmoid')(X)
#
#     model = Model(inputs = , outputs = )
#


