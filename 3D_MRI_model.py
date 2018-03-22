from nilearn import plotting, image
import numpy as np
import os
from os.path import join
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
import matplotlib.pyplot as plt
K.set_image_data_format('channels_last')


def batch_generator(files, n_epochs):
    while True:
        for mri in files:
            label = mri.split('_')[1]
            label = 0 if label == 'healthy' else 1

            img = image.load_img(mri).get_data()
            img = (img - np.mean(img))/np.std(img)
            x, y, z = img.shape
            yield (img.reshape(1, x, y, z, 1), np.array([label]))


def make_model():
    model = Sequential()

    #LAYER 1 (convolution 5x5x5)
    model.add(Conv3D(batch_input_shape=[None, 128, 128, 128, 1],
                     filters=16,
                     kernel_size=5,
                     strides=3,
                     padding='valid'))

    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))


    model.add(Conv3D(filters=32,
                     kernel_size=5,
                     strides=3,
                     padding='valid'))

    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))


    model.add(Flatten())

    model.add(Dense(200,
                    activation='relu'))


    model.add(Dense(100,
                    activation='relu'))

    model.add(Dense(1,
                    kernel_initializer='uniform',
                    activation='sigmoid'))

    print(model.output_shape)

    return model


data_path = "C:\\Users\\Larry\\NilearnStuff\\FinalDataset"
# data_path = 'EMC'
np.random.seed(10)

n_epochs = 50

files = [os.path.join(data_path, k) for k in os.listdir(data_path) if '_mri' in k]
perm_files = np.random.permutation(files)     #ONLY FIRST 200 DATAPOINTS


train_size = int(len(perm_files) * 0.7)
test_size = val_size = int(len(perm_files) * 0.15)

train_samples = perm_files[0: train_size]
val_samples = perm_files[train_size: train_size + val_size]
test_samples = perm_files[train_size + val_size: ]

opt = Adam(lr=0.01)
model = make_model()
model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(generator=batch_generator(train_samples, n_epochs=n_epochs),
                              steps_per_epoch=len(train_samples),
                              validation_data=batch_generator(val_samples, n_epochs=1),
                              validation_steps=len(val_samples),
                              verbose=1,
                              epochs=n_epochs)

model.save('3D_cnn_MRI_alldata.h5')

plt.plot(history.history['acc'])
plt.show()





