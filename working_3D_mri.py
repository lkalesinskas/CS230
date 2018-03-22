from nilearn import plotting, image
import numpy as np
import os
from os.path import join
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import LeakyReLU
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1_l2, l1, l2
from keras import backend as K
import matplotlib.pyplot as plt
K.set_image_data_format('channels_last')


def batch_generator(files, n_epochs):
    while True:
        perm = np.random.permutation(len(files))

        files = files[perm]
        for mri in files:
            label = mri.split('_')[1]
            label = 0 if label == 'healthy' else 1

            img = image.load_img(mri).get_data()
            img = (img - np.mean(img))/np.std(img)
            x, y, z = img.shape

            yield (img.reshape(1, x, y, z, 1), np.array([label]))


def make_model(l2_penalty):
    model = Sequential()

    #LAYER 1 (convolution 5x5x5)
    model.add(Conv3D(batch_input_shape=[None, 128, 128, 128, 1],
                     filters=16,
                     kernel_size=5,
                     strides=3,
                     padding='valid',
                     kernel_regularizer=l2(l2_penalty),
                     activation='relu'))


    model.add(MaxPooling3D(pool_size=(2, 2, 2)))


    model.add(Conv3D(filters=32,
                     kernel_size=5,
                     strides=3,
                     padding='valid',
                     activation = 'relu'))


    model.add(MaxPooling3D(pool_size=(2, 2, 2)))


    model.add(Flatten())

    model.add(Dense(100, activation = 'relu'))
    model.add(Dropout(rate=0.15))

    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(1,
                    kernel_initializer='uniform',
                    activation='sigmoid'))

    print(model.output_shape)

    return model


data_path = "C:\\Users\\Larry\\NilearnStuff\\FinalDataset"
#data_path = 'EMC'
np.random.seed(10)

n_epochs = 50

files = [os.path.join(data_path, k) for k in os.listdir(data_path) if '_mri' in k]
perm_files = np.random.permutation(files)


train_size = int(len(perm_files) * 0.7)
test_size = val_size = int(len(perm_files) * 0.15)

train_samples = perm_files[0: train_size]
val_samples = perm_files[train_size: train_size + val_size]
test_samples = perm_files[train_size + val_size: ]

for l2_penalty in [100, 10, 1, 0.1]:
    opt = Adam(lr=0.0001)
    model = make_model(l2_penalty)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit_generator(generator=batch_generator(train_samples, n_epochs=n_epochs),
                                  steps_per_epoch=len(train_samples),
                                  validation_data=batch_generator(val_samples, n_epochs=1),
                                  validation_steps=len(val_samples),
                                  verbose=1,
                                  epochs=n_epochs)

    plt.plot(history.history['acc'])
    filename = 'C:\\Users\\Larry\\NilearnStuff\\figure' + "_" + str(l2_penalty) + "_" + str(n_epochs) + ".png"
    plt.savefig(filename)
    plt.close()


#model.save('cnn_MRI_200.h5')


