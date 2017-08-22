from __future__ import print_function
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, merge, Reshape
from keras import backend as K
import numpy as np
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Convolution2D
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()



gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))


#print(tf.__version__)
print(keras.__version__)

import os
os.environ["KERAS_BACKEND"] = "theano"

import time
import glob


K.set_image_dim_ordering('tf')
time_i = time.time()

from keras.preprocessing.image import ImageDataGenerator
data_augmentation = True

batch_size = 32
num_classes = 2
num_epoch = 2000
pre_train_size = 200
filter_factor = 40
learning_rate_gen = 1e-4  # Warning: lr and decay vary across optimizers
learning_rate_dis = 1e-5  # Warning: lr and decay vary across optimizers
learning_rate_gan = 1e-5  # Warning: lr and decay vary across optimizers


decay_rate = 0.1
dropout_rate = 0.25
opti_id = 1  # [SGD, Adam, Adadelta, RMSprop]
loss_id = 0 # [mse, mae] # mse is always better



Dir0 = '../'
Dir1 = Dir0 + '../AllTrainTestSets/Encoder/'
#Dir0 = './'
#Dir1 = '/home/nes/Desktop/ConvNetData/lens/Encoder/'
Dir2 = ['single/', 'stack/'][1]
data_path = Dir1 + Dir2

DirOutType = ['noisy0', 'noisy1', 'noiseless']   # check above too

image_size = img_rows = img_cols = 45
num_channel = 1
num_files = 9000
train_split = 0.8   # 80 percent
num_train = int(train_split*num_files)


fileOut = 'GAN' + str(opti_id) + '_loss' + str(loss_id) + '_lrGen_' + str(learning_rate_gen)+ '_lrDis_' + str(learning_rate_dis) + '_lrGAN_' +str(learning_rate_gan) + '_decay' + str(decay_rate) + '_batch' + str(batch_size) + '_epoch' + str(num_epoch)


def load_train(fnames):
    img_data_list = []

    filelist = sorted(glob.glob(fnames + '/*npy'))[:num_files]
    for fileIn in filelist:  # restricting #files now [:num_files]
        # print(fileIn)
        img_data = np.load(fileIn)
        img_data = img_data[:28, :28]    # comment later
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

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


fnames = data_path + DirOutType[2]   #'noiseless'
noiseless_data, noiseless_target = load_train(fnames)
x_train = noiseless_data[0:num_train]
y_train = noiseless_target[0:num_train]
x_val = noiseless_data[num_train:num_files]
y_val = noiseless_target[num_train:num_files]

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)


# Build Generative model ...

def Generator():
#    filter_factor = 40

    g_input = Input(shape=[100])
    H = Dense( filter_factor*14*14, init='glorot_normal')(g_input)
    # H = BatchNormalization(mode=2)(H) # Commented by Nesar
    H = BatchNormalization()(H)
    H = Activation('relu')(H)
    H = Reshape( [14, 14, int(filter_factor)] )(H)
    H = UpSampling2D(size=(2, 2))(H)
    H = Convolution2D(3, 3, int(filter_factor/2), border_mode='same', init='glorot_uniform')(H)
    # H = BatchNormalization(mode=2)(H) # Commented by Nesar
    H = BatchNormalization()(H)
    H = Activation('relu')(H)
    H = Convolution2D(3, 3, int(filter_factor/4), border_mode='same', init='glorot_uniform')(H)
    # H = BatchNormalization(mode=2)(H) # Commented by Nesar
    H = BatchNormalization()(H)
    H = Activation('relu')(H)
    H = Convolution2D(1, 1, 1, border_mode='same', init='glorot_uniform')(H)
    g_V = Activation('sigmoid')(H)
    generator = Model(g_input,g_V)

    opt_gen = Adam(lr=learning_rate_gen, decay=decay_rate)
    generator.compile(loss='binary_crossentropy', optimizer=opt_gen)
    # generator.summary()

    return generator

# Build Discriminative model

def Discriminator():

    d_input = Input(shape=x_train.shape[1:])
    H = Convolution2D(5, 5, 256, subsample=(2, 2), border_mode = 'same', activation='relu')(d_input)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Convolution2D(5, 5, 512, subsample=(2, 2), border_mode = 'same', activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Flatten()(H)
    H = Dense(256)(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    d_V = Dense(2,activation='softmax')(H)
    discriminator = Model(d_input,d_V)

    opt_dis = Adam(lr=learning_rate_dis, decay=decay_rate)
    discriminator.compile(loss='categorical_crossentropy', optimizer= opt_dis)
    discriminator.summary()

    return discriminator

generator = Generator()
discriminator = Discriminator()


# Freeze weights in the discriminator for stacked training
make_trainable(discriminator, False)


# Build stacked GAN model
gan_input = Input(shape=[100])
H = generator(gan_input)
gan_V = discriminator(H)
GAN = Model(gan_input, gan_V)
opt_gan = Adam(lr=learning_rate_gan, decay=decay_rate)
GAN.compile(loss='categorical_crossentropy', optimizer= opt_gan)
GAN.summary()

####################################################################################################



# Pre-training
PreTrain = True
if PreTrain:
    ntrain = pre_train_size
    np.random.seed(123)
    trainidx = np.random.randint(0, x_train.shape[0], size = ntrain)
    x_train_selected = x_train[trainidx, :, :, :]

    # Pre-train the discriminator network ...
    noise_gen = np.random.uniform(0, 1, size=[x_train_selected.shape[0], 100])
    generated_images = generator.predict(noise_gen)
    x_stacked = np.concatenate((x_train_selected, generated_images))

    n = x_train_selected.shape[0]
    y_init = np.zeros([2 * n, 2])
    y_init[:n, 1] = 1
    y_init[n:, 0] = 1

    make_trainable(discriminator, True)
    discriminator.fit( x_stacked, y_init, nb_epoch=1, batch_size=32)
    y_hat = discriminator.predict(x_stacked)


    y_hat_idx = np.argmax(y_hat, axis=1)
    y_idx = np.argmax(y_init, axis=1)
    diff = y_idx - y_hat_idx
    n_tot = y_init.shape[0]
    n_rig = (diff == 0).sum()
    acc = n_rig * 100.0 / n_tot
    print("Accuracy: %0.02f pct (%d of %d) right" % (acc, n_rig, n_tot))



def plot_loss(losses):
        # display.clear_output(wait=True)
        # display.display(plt.gcf())
        plt.figure(figsize=(7,5))
        plt.plot(losses["dis"], label='discriminative loss')
        plt.plot(losses["gen"], label='generative loss')
        plt.legend()
        # plt.show()
        plt.savefig(Dir0 +'plots/loss'+fileOut +'.pdf')


def plot_gen(n_ex=16,dim=(4,4), figsize=(8,8) ):

    noise = np.random.uniform(0, 1, size=[n_ex, 100])
    generated_images = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        img = generated_images[i,:,:,0]
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    # plt.show()
    plt.savefig(Dir0 + 'plots/generated_images'+fileOut+'.pdf')




# set up loss storage vector
losses = {"dis":[], "gen":[]}



def train_for_n(nb_epoch=20, plt_frq=20, BATCH_SIZE=16):
    for e in (range(nb_epoch)):

        # Make generative images
        image_batch = x_train[np.random.randint(0, x_train.shape[0], size=BATCH_SIZE), :, :, :]
        noise_gen = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
        generated_images = generator.predict(noise_gen)

        # Train discriminator on generated images
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2 * BATCH_SIZE, 2])
        y[0:BATCH_SIZE, 1] = 1
        y[BATCH_SIZE:, 0] = 1

        make_trainable(discriminator, True)
        dis_loss = discriminator.train_on_batch(X, y)
        losses["dis"].append(dis_loss)

        # train Generator-Discriminator stack on input noise to non-generated output class
        noise_tr = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
        y2 = np.zeros([BATCH_SIZE, 2])
        y2[:, 1] = 1

        make_trainable(discriminator, False)
        gen_loss = GAN.train_on_batch(noise_tr, y2)
        losses["gen"].append(gen_loss)

        # Updates plots
        if e % plt_frq == plt_frq - 1:
            plot_loss(losses)
            plot_gen()


# K.set_value(opt.lr, 1e-5)
# K.set_value(dopt.lr, 1e-6)
train_for_n(nb_epoch= num_epoch, plt_frq= num_epoch, BATCH_SIZE=batch_size)

plot_gen(16, (4,4), (8,8))
plot_loss(losses)


def plot_real(n_ex=16, dim=(4, 4), figsize=(8, 8)):
    idx = np.random.randint(0, x_train.shape[0], n_ex)
    generated_images = x_train[idx, :, :, :]

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        img = generated_images[i, :, :, 0]
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(Dir0 +'plots/real_images'+fileOut+'.pdf')
    # plt.show()


plot_real()


# plt.show()

generator.save(Dir0+'../ModelOutGAN/GANGenerate_' + fileOut + '.hdf5')
discriminator.save(Dir0+'../ModelOutGAN/GANdiscriminate_' + fileOut + '.hdf5')
GAN.save(Dir0+'../ModelOutGAN/GAN_' + fileOut + '.hdf5')
# np.save('ModelOutEncode/Generate' + fileOut + '.npy', training_hist)



print (50 * '-')
time_j = time.time()
print(time_j - time_i, 'seconds')
print (50 * '-')
