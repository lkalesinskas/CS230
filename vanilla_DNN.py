from nilearn import plotting, image
import numpy as np
import os
from os.path import join
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import matplotlib.pyplot as plt
K.set_image_data_format('channels_last')


def batch_generator(files, n_epochs):
    while True:
        for mri in files:
            label = mri.split('_')[1]
            label = 0 if label == 'healthy' else 1

            img = image.load_img(mri).get_data()
            x, y, z = img.shape
            img = img[:, :, z//2]
            img = (img - np.mean(img))/np.std(img)
            yield (img.reshape(1, x * y), np.array([label]))


def make_model():
    model = Sequential()

    # model.add(MaxPooling3D(pool_size=(3, 3, 3),
    #                        batch_input_shape=[1, 128, 128, 1, 1]))
    # model.add(Flatten())


    model.add(Dense(128,
                    batch_input_shape=(None, 128 ** 2),
                    activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(128,
                    activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(128,
                    activation='relu'))

    # model.add(Dense(128))
    # model.add(Activation('relu'))

    model.add(Dense(1,
                    activation='sigmoid'))

    return model


data_path = "C:\\Users\\Larry\\NilearnStuff\\FinalDataset"
np.random.seed(10)

n_epochs = 10

files = [os.path.join(data_path, k) for k in os.listdir(data_path) if '_mri' in k]
perm_files = np.random.permutation(files)     #ONLY FIRST 200 DATAPOINTS


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
                              validation_data=batch_generator(val_samples, n_epochs=1),
                              validation_steps=len(val_samples),
                              verbose=1,
                              epochs=n_epochs)

plt.plot(history.history['acc'])
plt.show()


model.save('dnn_MRI_200.h5')

