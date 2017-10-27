from __future__ import print_function

import numpy as np
# get_ipython().magic(u'matplotlib inline')
# import matplotlib
import matplotlib.pyplot as plt


# In[9]:

import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout
from keras import backend as K
import numpy as np
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.utils import np_utils


K.set_image_dim_ordering('tf')
import time
time_i = time.time()
import glob

from keras.preprocessing.image import ImageDataGenerator
data_augmentation = True

batch_size = 4
num_classes = 2
epochs = 3


Dir1 = '/home/nes/Desktop/ConvNetData/lens/Encoder/'
Dir2 = ['single/', 'stack/'][1]
data_path = Dir1 + Dir2
# names = ['lensed', 'unlensed']
# data_dir_list = ['lensed_outputs', 'unlensed_outputs']

DirOutType = ['noisy0', 'noisy1', 'noiseless'][0]   # check above too


image_size = img_rows = img_cols = 45
num_channel = 1
# num_classes = 2
num_files = 800*num_classes
train_split = 0.8   # 80 percent
num_train = int(train_split*num_files)

# (x_train1, y_train1), (x_test1, y_test1) = mnist.load_data()

def load_train(fnames):
    img_data_list = []


# for labelID in [0, 1]:
		# name = names[labelID]
		# for img_ind in range( int(num_files / num_classes) ):
        #
			# input_img = np.load(data_path + '/' + name + '_outputs/' + name + str(img_ind) + '.npy')
			# if np.isnan(input_img).any():
			# 	print (labelID, img_ind, ' -- ERROR: NaN')
			# else:
			# 	img_data_list.append(input_img)

    filelist = sorted(glob.glob(fnames))
    for fileIn in filelist[:num_files]:
        print(fileIn)
        img_data = np.load(fileIn)
        # print(fileIn)
        img_data = np.array(img_data)
        img_data = img_data.astype('float32')

        img_data /= 255.
        # print (img_data.shape)
        img_data_list.append(img_data)
        print(np.array(img_data_list).shape)

# 	if num_channel == 1:
# 		if K.image_dim_ordering() == 'th':
# 			img_data = np.expand_dims(img_data, axis=1)
# 			print (img_data.shape)
# 		else:
# 			img_data = np.expand_dims(img_data, axis=4)
# 			print (img_data.shape)
# 	else:
# 		if K.image_dim_ordering() == 'th':
# 			img_data = np.rollaxis(img_data, 3, 1)
# 	print (img_data.shape)

	X_train = np.array(img_data_list)
    labels = np.load(data_path + DirOutType +'_5para.npy')
    # print (labels.shape)

    y_train = np_utils.to_categorical(labels[:,0], num_classes)

    np.random.seed(12345)
    shuffleOrder = np.arange(X_train.shape[0])

    np.random.shuffle(shuffleOrder)
    X_train = X_train[shuffleOrder]
    y_train = y_train[shuffleOrder]


    return X_train, y_train

def read_and_normalize_train_data():
    train_data, train_target = load_train()
    train_data = np.array(train_data, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.float32)
    m = train_data.mean()
    s = train_data.std()

    print ('Train mean, sd:', m, s )
    train_data -= m
    train_data /= s
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target

# train_data, train_target = read_and_normalize_train_data()

train_data, train_target = load_train(data_path + 'noiseless' + '/*npy')

x_train = train_data[0:num_train,:,:]
y_train = train_target[0:num_train]

x_test = train_data[num_train:num_files,:,:]
y_test = train_target[num_train:num_files]


print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

plt.imshow(x_train[100], cmap='gray')
print(x_train.max(), x_train.min())


x_train = x_train.reshape(x_train.shape[0], image_size*image_size)
x_test = x_test.reshape(x_test.shape[0], image_size*image_size)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# ### Build and train an autoencoder

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
# encoding_dim = 2   #  2 floats -> compression of factor 392


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

autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

history = autoencoder.fit(x_train, x_train,
                          batch_size=batch_size, epochs=epochs,
                          verbose=1, validation_data=(x_test, x_test))

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)



n = 4  # how many digits we will display
fig = plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(image_size, image_size))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(image_size, image_size))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# ### Denoising autoencoder

noisy_train_data, noisy_train_target = load_train(data_path + 'noisy1' + '/*npy')


# noise_factor = 0.01
# x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
# x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
#
# x_train_noisy = np.clip(x_train_noisy, 0., 1.)
# x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# print(x_train_noisy.shape)





x_train_noisy = noisy_train_data[0:num_train,:,:]
y_train_noisy = noisy_train_target[0:num_train]

x_test_noisy = noisy_train_data[num_train:num_files,:,:]
y_test_noisy = noisy_train_target[num_train:num_files]


# print('x_train shape:', x_train.shape)
# print('y_train shape:', y_train.shape)

# plt.imshow(x_train[100], cmap='gray')
# print(x_train.max(), x_train.min())


x_train_noisy = x_train_noisy.reshape(x_train.shape[0], image_size*image_size)
x_test_noisy = x_test_noisy.reshape(x_test.shape[0], image_size*image_size)
x_train_noisy = x_train_noisy.astype('float32')
x_test_noisy = x_test_noisy.astype('float32')
x_train_noisy /= 255
x_test_noisy /= 255
print(x_train_noisy.shape[0], 'train samples')
print(x_test_noisy.shape[0], 'test samples')



autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

history = autoencoder.fit(x_train_noisy, x_train,
                          batch_size=batch_size, epochs=epochs,
                          verbose=1, validation_data=(x_test_noisy, x_test))



encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 8  # how many digits we will display
fig = plt.figure(figsize=(12, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(image_size, image_size))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(image_size, image_size))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()










# In[84]:

TestColumn = False
if TestColumn:

    # what about something more drastic?
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_one_column = np.zeros(x_train.shape)
    x_test_one_column = np.zeros(x_test.shape)

    x_train_one_column[:, :, 14] = x_train[:, :, 14]
    x_test_one_column[:, :, 14] = x_test[:, :, 14]

    x_train = x_train.reshape(x_train.shape[0], image_size).astype('float32') / 255.
    x_test = x_test.reshape(x_train.shape[0], image_size).astype('float32') / 255.

    x_train_one_column = x_train_one_column.reshape(x_train.shape[0], image_size).astype('float32') / 255.
    x_test_one_column = x_test_one_column.reshape(x_test.shape[0], image_size).astype('float32') / 255.


    # In[79]:

    autoencoder.fit(x_train_one_column, x_train,
                    batch_size=batch_size, epochs=5,
                    verbose=1, validation_data=(x_test_one_column, x_test))


    # In[46]:

    encoded_imgs = encoder.predict(x_test_one_column)
    decoded_imgs = decoder.predict(encoded_imgs)

    n = 10  # how many digits we will display
    fig = plt.figure(figsize=(20, 6))
    for i in range(10):
        # display original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display original
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(x_test_one_column[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + n +n )
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

