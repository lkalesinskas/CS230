"""
Initial model training on multiple (10) slices of the MRI data
Only 1 dataset (58 samples 29 (+), 29(-)) was used
"""

from nilearn import plotting, image
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import matplotlib.pyplot as plt
from keras import regularizers
K.set_image_data_format('channels_last')
import os
import pandas as pd


def batch_generator(files, n_epochs):
    while True:
        for mri in files:
            label = mri.split('_')[1]
            label = 0 if label == 'healthy' else 1

            img = image.load_img(mri).get_data()
            x, y, z = img.shape
            img = img[:, :, z//2]
            yield (img.reshape(1, x, y, 1), np.array([label]))


def make_model():
    """
    Create model for 2D convolution with slices as channels
    :return: created Keras model
    """
    model = Sequential()

    #LAYER 1 (convolution 5x5)
    model.add(Conv2D(batch_input_shape=[1, 128, 128, 1],
                     filters=16,
                     kernel_size=5,
                     strides=3,
                     padding='same',
                     kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    #LAYER 2 (convolution 3x3)
    model.add(Conv2D(filters=32,
                     kernel_size=5,
                     strides=1,
                     padding='same',
                     kernel_regularizer=regularizers.l2(0.01) ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    #LAYER 3(fully connected)
    model.add(Flatten())
    model.add(Dense(128,
                    activation='relu'))
    model.add(Dropout(rate=0.2))


    #LAYER 4(output)
    model.add(Dense(1,
                    kernel_initializer='uniform',
                    activation='sigmoid'))

    return model


model = make_model()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

data_path = "C:\\Users\\Larry\\NilearnStuff\\FinalDataset"
#data_path = 'EMC'
np.random.seed(10)

n_epochs = 10

files = [os.path.join(data_path, k) for k in os.listdir(data_path) if '_mri' in k]
perm_files = np.random.permutation(files)#[:200]     #ONLY FIRST 200 DATAPOINTS


train_size = int(len(perm_files) * 0.7)
test_size = val_size = int(len(perm_files) * 0.15)

train_samples = perm_files[0: train_size]
val_samples = perm_files[train_size: train_size + val_size]
test_samples = perm_files[train_size + val_size:]


model = make_model()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(generator=batch_generator(train_samples, n_epochs=n_epochs),
                              steps_per_epoch=len(train_samples),
                              validation_data=batch_generator(val_samples, n_epochs=1),
                              validation_steps=len(val_samples),
                              verbose=1,
                              epochs=n_epochs)

plt.plot(history.history['acc'])
plt.show()


# model.save('cnn_MRI_1slice_200.h5')
