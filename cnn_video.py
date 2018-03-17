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

data_path = 'EMC/'


np.random.seed(10)
def batch_generator(n=1):
    files = [os.path.join(data_path, k) for k in os.listdir(data_path) if 'fmri' in k]
    files = np.random.permutation(files)
    labels = pd.read_csv('EMC_labels.csv', index_col='SUB_ID').DX_GROUP
    for fmri in files:
        sub_id = int(fmri.split('/')[1].split('_')[0])
        img = image.load_img(fmri).get_data()
        x, y, z, Tx = img.shape
        zs = [int(0.25 * z), int(0.5 * z), int(0.75 * z)]
        img = img[:, :, zs, :]    #get the center slice on the z axis
        label = abs(labels.loc[sub_id] - 2)
        print(label)
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

    model.add(Conv3D(filters=16,
                     kernel_size=5,
                     strides=1,
                     padding='valid'))

    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3)))

    print('layer 2 output:', model.output_shape)

    model.add(Flatten())

    model.add(Dense(1,
                    kernel_initializer='uniform',
                    activation='sigmoid'))

    return model

files = [os.path.join(data_path, k) for k in os.listdir(data_path) if 'fmri' in k]
model = make_model()
opt = Adam(learning_rate=)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit_generator(generator=batch_generator(),
                              steps_per_epoch=len(files),
                              verbose=1)
model.save('cnn_video_EMC.h5')
plt.plot(history.history['acc'])
plt.show()
