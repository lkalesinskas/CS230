

from nilearn import plotting, image
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from os.path import join
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.optimizers import Adam
from keras import regularizers
import matplotlib.pyplot as plt
K.set_image_data_format('channels_last')


def batch_generator(files, n_epochs):
    files = np.array(files)
    ep = 0
    while True:
        if ep % 5 == 0:
            perm = np.random.permutation(len(files))
            files = files[perm]
        for mri in files:
            label = mri.split('_')[1]
            label = 0 if label == 'healthy' else 1

            img = image.load_img(mri).get_data()
            img = (img - np.mean(img))/np.std(img)
            x, y, z = img.shape
            yield (img.reshape(1, x, y, z, 1), np.array([label]))
        ep += 1


def make_model():
    model = Sequential()

    #LAYER 1 (convolution 5x5x5)
    model.add(Conv3D(batch_input_shape=[None, 128, 128, 128, 1],
                     filters=30,
                     kernel_size=5,
                     strides=3,
                     padding='valid', kernel_regularizer=regularizers.l2(0.01)))

    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Dropout(rate=0.1))


    model.add(Conv3D(filters=32,
                     kernel_size=5,
                     strides=3,
                     padding='valid'))

    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Dropout(rate=0.11))

    model.add(Flatten())
    model.add(Dense(300,
                    activation='relu'))
    model.add(Dropout(rate=0.17))

    model.add(Dense(100,
                     activation='relu'))

    model.add(Dense(1,
                    kernel_initializer='uniform',
                    activation='sigmoid'))

    return model


def metrics(preds, labels):
    preds = list(preds.reshape(-1))
    pred_labels = [1 if k>= 0.5 else 0 for k in preds]
    acc = accuracy_score(labels, pred_labels)
    p, r, f1, sup = precision_recall_fscore_support(labels, pred_labels)
    return acc, p, r, f1, sup


data_path = "C:\\Users\\Larry\\NilearnStuff\\FinalDataset"
# data_path = 'EMC'
np.random.seed(10)

n_epochs = 1
accs = []
n_folds = 5

files = [os.path.join(data_path, k) for k in os.listdir(data_path) if '_mri' in k]
perm_files = np.random.permutation(files)
train_size = int(len(perm_files) * 0.7)
test_size = val_size = int(len(perm_files) * 0.15)
test_samples = perm_files[-test_size:]
perm_files = list(perm_files[:-test_size])

for k in range(n_folds):
    val_samples = perm_files[k * (val_size): (k + 1) * val_size]
    train_samples = perm_files[:k * (val_size)] + perm_files[:(k + 1) * val_size]

    print('Starting fold', k)

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

    val_predictions = model.predict_generator(generator=batch_generator(val_samples, n_epochs=1),
                                              steps=len(val_samples),
                                              verbose=0)

    val_labels = [k.split('_')[1] for k in val_samples]
    val_labels = [1 if k == 'autism' else 0 for k in val_labels]
    a, p, r, f1, _ = metrics(val_predictions, val_labels)
    print('Validation Metrics on fold', k)
    print('precision', p)
    print('recall', r)
    print('f1 score', f1)
    print('accuracy', a)

    test_predictions = model.predict_generator(generator=batch_generator(test_samples, n_epochs=1),
                                               steps=len(test_samples),
                                               verbose=0)

    test_labels = [k.split('_')[1] for k in test_samples]
    test_labels = [1 if k == 'autism' else 0 for k in test_labels]
    a, p, r, f1, _ = metrics(test_predictions, test_labels)
    print('Test Metrics')
    print('precision', p)
    print('recall', r)
    print('f1 score', f1)
    print('accuracy', a)

    train_acc = history.history['acc']
    val_acc = history.history['val_acc']

    np.savetxt("C:\\Users\\Larry\\NilearnStuff\\3D_MRI_NWdeep_TRAIN_fold_{}.txt".format(k), train_acc)
    np.savetxt("C:\\Users\\Larry\\NilearnStuff\\3D_MRI_NWdeep_VALIDATION_fold_{}.txt".format(k), val_acc)

    accs.append(history.history['acc'][-1])

print('fold accuracies', accs)

