from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Input, Conv3D, MaxPooling3D, LSTM, TimeDistributed
from keras import backend as K
from nilearn import plotting, image
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
K.set_image_data_format('channels_first')

data_path =  "C:\\Users\\Larry\\NilearnStuff\\FinalDataset"


np.random.seed(10)
def batch_generator(files, n_epochs):
    while True:
        for fmri in files:
            label = fmri.split('_')[1]
            label = 0 if label == 'healthy' else 1

            img = image.load_img(fmri).get_data()
            x, y, z, Tx = img.shape
            zs = [int(0.25 * z), int(0.5 * z), int(0.75 * z)]
            img = img[:, :, zs, :]    #get the center slice on the z axis

            yield (img.reshape(1, 3, x, y, Tx), np.array([label]))

def make_model():
    model = Sequential()

    #LAYER 1 (convolution 5x5x5)
    model.add(Conv3D(batch_input_shape=[None, 3, 64, 64, 85],
                     filters=32,
                     kernel_size=5,
                     strides=1,
                     padding='same'))

    model.add(Activation('relu'))
    # model.add(MaxPooling3D(pool_size=(5, 5, 5)))

    print('layer 1 output:', model.output_shape)

    model.add(Conv3D(filters=64,
                     kernel_size=5,
                     strides=1,
                     padding='valid'))

    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3)))

    print('layer 2 output:', model.output_shape)

    model.add(Flatten())

    model.add(Dense(128,
                    activation='relu'))


    model.add(Dense(1,
                    kernel_initializer='uniform',
                    activation='sigmoid'))

    return model


n_epochs=5
files = [os.path.join(data_path, k) for k in os.listdir(data_path) if 'fmri' in k]
perm_files = np.random.permutation(files)

train_size = int(len(perm_files) * 0.7)
test_size = val_size = int(len(perm_files) * 0.15)

train_samples = perm_files[0: train_size]
val_samples = perm_files[train_size: train_size + val_size]
test_samples = perm_files[train_size + val_size: ]


model = make_model()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])



history = model.fit_generator(generator=batch_generator(train_samples, n_epochs=n_epochs),
                              steps_per_epoch=len(train_samples),
                              validation_data=batch_generator(val_samples, n_epochs=n_epochs),
                              validation_steps=len(val_samples),
                              verbose=1,
                              epochs=n_epochs)
model.save('cnn_video_alldata.h5')
plt.plot(history.history['acc'])
plt.show()


#
#
#
# files = [os.path.join(data_path, k) for k in os.listdir(data_path) if 'fmri' in k]
# model = make_model()
#
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# history = model.fit_generator(generator=batch_generator(),
#                               steps_per_epoch=len(files),
#                               verbose=1)
# model.save('cnn_video_EMC.h5')
# plt.plot(history.history['acc'])
# plt.show()
