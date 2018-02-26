"""
Initial model training on multiple (10) slices of the MRI data
Only 1 dataset (58 samples 29 (+), 29(-)) was used
"""

from nilearn import plotting, image
import numpy as np
import os
import pandas as pd

# Read the labels file and extract relevant information
# 1 = Autism, 2 = Control
df = pd.read_csv('/Users/rohanpaul/Downloads/ABIDEII-BNI_1.csv')
df = df[['SUB_ID', 'DX_GROUP']]
df = df.set_index('SUB_ID')

# Get file paths for images
path = '/Users/rohanpaul/Downloads/ABIDEII-BNI_1/'
folders = [k for k in os.listdir(path) if not k.startswith('.DS')]
fpaths = []
for folder in folders:
    fpath = os.path.join(path, folder, 'session_1/anat_1/anat.nii.gz')
    fpaths.append((folder, fpath))

# extract slices from each example and separate into autism and healthy
aut_mri, control_mri = [], []
for item in fpaths:
    if df.loc[int(item[0])].DX_GROUP == 2:
        control_mri.append(image.load_img(item[1]).get_data()[:, :, 50:200:15])  #get 10 slices between index z=50 and 200
    elif df.loc[int(item[0])].DX_GROUP == 1:
        aut_mri.append(image.load_img(item[1]).get_data()[:,:,50:200:15])
    else:
        raise Exception


labels = np.array([1 for k in range(len(aut_mri))] + [0 for k in range(len(control_mri))])
X = np.array(aut_mri + control_mri)


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
K.set_image_data_format('channels_last')


def first_model():
    """
    Create model for 2D convolution with slices as channels
    :return: created Keras model
    """
    model = Sequential()

    #LAYER 1 (convolution 5x5)
    model.add(Conv2D(input_shape=[193, 256, 10],
                     filters=16,
                     kernel_size=5,
                     strides=3,
                     padding='same',
                     kernel_initializer='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(5, 5)))


    #LAYER 2 (convolution 3x3)
    model.add(Conv2D(filters=32,
                     kernel_size=5,
                     strides=3,
                     padding='same',
                     kernel_initializer='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(5, 5)))


    #LAYER 3(fully connected)
    model.add(Flatten())
    model.add(Dense(model.output_shape[1],
                    kernel_initializer='glorot_uniform',
                    activation='relu'))
    model.add(Dropout(rate=0.2))


    #LAYER 4(output)
    model.add(Dense(1,
                    kernel_initializer='uniform',
                    activation='sigmoid'))

    return model


model = first_model()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


np.random.seed(0)
#Create train/test split of 80/20
train_idx = np.random.randint(0, len(X), size=4*len(X)//5)
test_idx = np.array([k for k in range(len(X)) if k not in train_idx])

permute_train = np.random.permutation(train_idx)
Xtrain = X[permute_train].reshape((len(train_idx), 193, 256, 10))
ytrain = labels[permute_train]


permute_test = np.random.permutation(test_idx)
Xtest = X[permute_test].reshape((len(test_idx), 193, 256, 10))
ytest = labels[permute_test]

#fit and evaluate model
history = model.fit(Xtrain, ytrain, epochs=10, batch_size=1, verbose=1)
test_metric = model.evaluate(Xtest, ytest, batch_size=1, verbose=1)


print('metrics:', test_metric)