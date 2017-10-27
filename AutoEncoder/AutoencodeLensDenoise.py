"""
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
learning_rate = 0.001  # Warning: lr and decay vary across optimizers
decay_rate = 0.1
opti_id = 1  # [SGD, Adam, Adadelta, RMSprop]
loss_id = 0 # [mse, mae] # mse is always better


Dir1 = '/home/nes/Desktop/ConvNetData/lens/Encoder/'
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

fnames = data_path + DirOutType[1]  #noisy1
noisy_data, noisy_target = load_train(fnames)
x_train_noisy = noisy_data[0:num_train]
y_train_noisy = noisy_target[0:num_train]
x_val_noisy = noisy_data[num_train:num_files]
y_val_noisy = noisy_target[num_train:num_files]

plotCheck = False
if plotCheck:

    fig, ax = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(10, 5))
    plt.suptitle(DirOutType[2])

    count = 0
    np.random.seed(1234)
    indx = np.random.randint(20, size = 8)

    for ind in indx:
        pixel = x_train[ind].reshape(image_size, image_size)
        # for i in range(numPlots):
        ax[count / 4, count % 4].imshow(pixel, cmap=plt.get_cmap('gray'))
        ax[count / 4, count % 4].set_title(str(ind))

        count += 1


    fig, ax = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(10, 5))
    plt.suptitle(DirOutType[1])

    count = 0
    for ind in indx:
        pixel = x_train_noisy[ind].reshape(image_size, image_size)
        # for i in range(numPlots):
        ax[count / 4, count % 4].imshow(pixel, cmap=plt.get_cmap('gray'))
        ax[count / 4, count % 4].set_title(str(ind))

        count += 1

# ### Build and train an autoencoder

# this is the size of our encoded representations
encoding_dim = 64  # 32 floats -> compression of factor 32.6, assuming the input is 2025 floats

def AutoModel_orig():

    # this is our input placeholder
    input_img = Input(shape=(image_size*image_size,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(image_size*image_size, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(inputs=input_img, outputs=decoded)
    print("autoencoder model created")


    # this model maps an input to its encoded representation
    encoder = Model(inputs=input_img, outputs=encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))

    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]

    # create the decoder model
    decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))

    adam = Adam(lr=learning_rate, decay=decay_rate)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder


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



def AutoModel_conv():

    # this is our input placeholder
    input_img = Input(shape=(image_size,image_size,1))

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (6, 6, 8) i.e. 288-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = ZeroPadding2D( ((1, 0), (1, 0) ) )(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)



    autoencoder = Model(inputs=input_img, outputs=decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    # this model maps an input to its reconstruction
    print("autoencoder model created")

    # this model maps an input to its encoded representation
    encoder = Model(inputs=input_img, outputs=encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoding_dim =  (6,6,8)
    encoded_input = Input(shape=encoding_dim)

    # retrieve the last layer of the autoencoder model
    # decoder_layer = autoencoder.layers[-1]
    deco = autoencoder.layers[-8](encoded_input)
    deco = autoencoder.layers[-7](deco)
    deco = autoencoder.layers[-6](deco)
    deco = autoencoder.layers[-5](deco)
    deco = autoencoder.layers[-4](deco)
    deco = autoencoder.layers[-3](deco)
    deco = autoencoder.layers[-2](deco)
    deco = autoencoder.layers[-1](deco)
    # create the decoder model
    decoder = Model(encoded_input, deco)


    # create the decoder model
    # decoder = Model(inputs=encoded_input, outputs= autoencoder.output  )

    # adam = Adam(lr=learning_rate, decay=decay_rate)
    # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder





encoder, decoder, autoencoder = AutoModel_conv()
# autoencoder.summary()

# Denoising autoencoder
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

ModelFit = autoencoder.fit(x_train_noisy, x_train,
                          batch_size=batch_size, epochs= num_epoch,
                          verbose=2, validation_data=(x_val_noisy, x_val))


SaveModel = True
if SaveModel:
    epochs = np.arange(1, num_epoch+1)
    train_loss = ModelFit.history['loss']
    val_loss = ModelFit.history['val_loss']

    training_hist = np.vstack([epochs, train_loss, val_loss])


    fileOut = 'Stack_opti' + str(opti_id) + '_loss' + str(loss_id) + '_lr' + str(learning_rate) + '_decay' + str(decay_rate) + '_batch' + str(batch_size) + '_epoch' + str(num_epoch)

    autoencoder.save('ModelOutEncode/autoDenoise_' + fileOut + '.hdf5')
    encoder.save('ModelOutEncode/encodeDenoise_' + fileOut + '.hdf5')
    decoder.save('ModelOutEncode/decodeDenoise_' + fileOut + '.hdf5')
    np.save('ModelOutEncode/Denoise'+fileOut+'.npy', training_hist)

time_j = time.time()
print(time_j - time_i, 'seconds')

#-----------------------------------------------------------------------------------