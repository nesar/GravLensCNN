"""
Author: Nesar Ramachandra 

Only train over lensed images now -- so o/p also seems lensed
Have to train over everything!!!


check again if noiseless and noisy images are matching!
Check 1d/2d issue
Convolutional encoding


increase number of features per layer

"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout
from keras import backend as K
import numpy as np
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.utils import np_utils
import time
import glob

time_i = time.time()
K.set_image_dim_ordering('tf')

from keras.preprocessing.image import ImageDataGenerator
data_augmentation = True

batch_size = 32
num_classes = 2
num_epoch = 200
learning_rate = 0.00005  # Warning: lr and decay vary across optimizers
decay_rate = 0.0
opti_id = 1  # [SGD, Adam, Adadelta, RMSprop]
loss_id = 0 # [mse, mae] # mse is always better


Dir1 = '../../AllTrainTestSets/Encoder/'
Dir2 = ['single/', 'stack/'][1]
data_path = Dir1 + Dir2

DirOutType = ['noisy0', 'noisy1', 'noiseless']   # check above too

image_size = img_rows = img_cols = 45
num_channel = 1
num_files = 9000
train_split = 0.8   # 80 percent
num_train = int(train_split*num_files)

def load_train(fnames):
    img_data_list = []

    filelist = sorted(glob.glob(fnames + '/*npy'))
    for fileIn in filelist:  # restricting #files now [:num_files]
        # print(fileIn)
        img_data = np.load(fileIn)
        # print(fileIn)
        ravelTrue = False
        if ravelTrue: img_data = np.ravel(np.array(img_data))
        img_data = img_data.astype('float32')

        img_data /= 255.
        expandTrue = True
        if expandTrue: img_data = np.expand_dims(img_data, axis=4)

        # print (img_data.shape)
        img_data_list.append(img_data)
        # print(np.array(img_data_list).shape)

    X_train = np.array(img_data_list)
    # labels = np.load(fnames +'_5para.npy')[:num_files]
    print (X_train.shape)
    labels = np.ones([X_train.shape[0], ])

    y_train = np_utils.to_categorical(labels, num_classes)

    np.random.seed(12345)
    shuffleOrder = np.arange(X_train.shape[0])

    # np.random.shuffle(shuffleOrder)
    # print(shuffleOrder)

    X_train = X_train[shuffleOrder]
    y_train = y_train[shuffleOrder]

    return X_train, y_train

fnames = data_path + DirOutType[2]   #'noiseless'
noiseless_data, noiseless_target = load_train(fnames)
x_train = noiseless_data[0:num_train]
y_train = noiseless_target[0:num_train]
x_val = noiseless_data[num_train:num_files]
y_val = noiseless_target[num_train:num_files]

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

# this is the size of our encoded representations
encoding_dim = 64  # 32 floats -> compression of factor 32.6, assuming the input is 2025 floats


def AutoModel_deep():

    # this is our input placeholder
    input_img = Input(shape=(image_size*image_size,))
    # "encoded" is the encoded representation of the input
    # encoded = Dense(encoding_dim, activation='relu')(input_img)
    # # "decoded" is the lossy reconstruction of the input
    # decoded = Dense(image_size*image_size, activation='sigmoid')(encoded)

    encoded1 = Dense(128, activation='relu')(input_img)
    encoded2 = Dense(64, activation='relu')(encoded1)
    encoded3 = Dense(32, activation='relu')(encoded2)

    decoded1 = Dense(64, activation='relu')(encoded3)
    decoded2 = Dense(128, activation='relu')(decoded1)
    decoded3 = Dense(image_size*image_size, activation='sigmoid')(decoded2)


    # this model maps an input to its reconstruction
    autoencoder = Model(inputs=input_img, outputs=decoded3)
    print("autoencoder model created")


    # this model maps an input to its encoded representation
    encoder = Model(inputs=input_img, outputs=encoded3)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))

    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]

    # create the decoder model
    decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))

    adam = Adam(lr=learning_rate, decay=decay_rate)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder


encoder, decoder, autoencoder = AutoModel_deep()
# autoencoder.summary()

# Denoising autoencoder
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

ModelFit = autoencoder.fit(x_train, x_train ,
                          batch_size=batch_size, epochs= num_epoch,
                          verbose=2, validation_data=(x_val, x_val))
