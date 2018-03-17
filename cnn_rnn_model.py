from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Input, Conv3D, MaxPooling3D, LSTM, TimeDistributed
from keras import backend as K
from nilearn import plotting, image
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
K.set_image_data_format('channels_first')

data_path = "C:\\Users\\Larry\\NilearnStuff\\FinalDataset"


np.random.seed(10)
def batch_generator(n=1):
    files = [os.path.join(data_path, k) for k in os.listdir(data_path) if 'fmri' in k]
    files = np.random.permutation(files)
    for fmri in files:
        label = fmri.split('_')[1]
        label = 0 if label == 'healthy' else 1
        img = image.load_img(fmri).get_data()
        x, y, z, Tx = img.shape
        yield (img.reshape(1, Tx, 1, x, y, z), np.array([label]))


def make_model(input_shape):
    model = Sequential()
    Tx, x, y, z = input_shape

    model.add(TimeDistributed(Conv3D(filters=16,
                                     kernel_size=5,
                                     strides=1,
                                     padding='same'),
                              batch_input_shape=[1, Tx, 1, x, y, z]))
    model.add(Activation('relu'))
    model.add(TimeDistributed(MaxPooling3D(pool_size=(2, 2, 2))))

    model.add(TimeDistributed(Conv3D(filters=32,
                                     kernel_size=3,
                                     strides=1,
                                     padding='valid')))
    model.add(Activation('relu'))
    model.add(TimeDistributed(MaxPooling3D(pool_size=(3, 3, 3))))

    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(1, batch_input_shape=(None, 100),  activation='sigmoid'))

    return model



files = [os.path.join(data_path, k) for k in os.listdir(data_path) if 'fmri' in k]
model = make_model((85, 64, 64, 31))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit_generator(generator=batch_generator(),
                              steps_per_epoch=len(files),
                              verbose=1,
                              epochs=5)
model.save('cnn_rnn_fulldata.h5')
plt.plot(history.history['acc'])
plt.show()

