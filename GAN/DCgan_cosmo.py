from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math

from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam, Adadelta
from keras import backend as K
#K.set_image_dim_ordering('tf')
import time
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pylab as plt


class LensData:
    '''
        Usage - lensData().load_data

        1) input_images: loads npy files, - shuffled randomly.
        2) normalize_data: re-scaling (-1, 1)
        3) load_data: returns 2 tuples: (x_train, y_train), (x_test, y_test)
            where y_train and y_test are already one-hot-encoded (using keras.np_utils)
    '''
    def __init__(self, num_classes = 2, num_channel = 1, train_val_split = 0.8, files_per_class =
    8000):

        self.num_channel = num_channel
        self.num_classes = num_classes
        self.files_per_class = files_per_class
        self.num_files = self.files_per_class*self.num_classes
        self.train_val_split = train_val_split
        self.num_train = int(self.train_val_split*self.num_files)

        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []


    def input_images(self):
        img_data_list = []
        labels = []

        file_idx = np.arange(int(self.num_files / self.num_classes))
        np.random.seed(444)
        np.random.shuffle(file_idx)

        for img_ind in file_idx:
            for labelID in [0, 1]:
                name = names[labelID]

                # print(labelID)
                input_img = np.load(data_path + '/' + name + '_outputs/' + name + str(img_ind) + '.npy')
                if np.isnan(input_img).any():
                    print (labelID, img_ind, ' -- ERROR: NaN')
                else:
                    img_data_list.append(input_img)
                    labels.append(labelID)

        img_data = np.array(img_data_list)
        img_data = img_data.astype('float32')


        labels = np.array(labels)
        labels = labels.astype('int')


        img_data /= 255.
        #print (img_data.shape)

        if self.num_channel == 1:
            if K.image_dim_ordering() == 'th':
                img_data = np.expand_dims(img_data, axis=1)
                #print (img_data.shape)
            else:
                img_data = np.expand_dims(img_data, axis=4)
                #print (img_data.shape)
        else:
            if K.image_dim_ordering() == 'th':
                img_data = np.rollaxis(img_data, 3, 1)
        #print (img_data.shape)


        self.train_data = img_data
        # labels = np.load(Dir1 + Dir2 + Dir3 + 'Train5para.npy')
        print (labels)

        self.train_target = np_utils.to_categorical(labels, self.num_classes)

        return self.train_data, self.train_target


    def normalize_data(self):
        train_data, train_target = self.input_images()
        train_data = np.array(train_data, dtype=np.float32)
        train_target = np.array(train_target, dtype=np.float32)
        m = train_data.mean()
        s = train_data.std()

        print ('Train mean, sd:', m, s)
        train_data -= m
        train_data /= s
        #print('Train shape:', train_data.shape)
        #print(train_data.shape[0], 'train samples')
        return train_data, train_target


    def load_data(self):
        train_data, train_target = self.normalize_data()


        np.random.seed(1234)
        shuffleOrder = np.arange(self.train_data[0:self.num_train, :, :, :].shape[0])
        np.random.shuffle(shuffleOrder)

        self.X_train = train_data[0:self.num_train, :, :, :][shuffleOrder]
        self.y_train = train_target[0:self.num_train][shuffleOrder]

        shuffleOrder = np.arange(self.train_data[self.num_train:self.num_files, :, :, :].shape[0])
        np.random.shuffle(shuffleOrder)

        self.X_test = train_data[self.num_train:self.num_files, :, :, :][shuffleOrder]
        self.y_test = train_target[self.num_train:self.num_files][shuffleOrder]

        return (self.X_train, self.y_train), (self.X_test, self.y_test)

        #return (self.X_train[:,:,:,:], self.y_train), (self.X_test[:,0:28, :,0:28], self.y_test)






def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(
            Conv2D(64, (5, 5),
            padding='same',
            input_shape=(28, 28, 1))
            )
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image


def train(BATCH_SIZE):
#    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    lens = LensData()
    (X_train, y_train), (X_test, y_test) = lens.load_data()
   

    #print(20*'~1')
    #print(X_train.shape)    

    X_train = (X_train.astype(np.float32) - 127.5)/127.5   
    X_train = X_train[:, 0:28, 0:28, 0]
    

    #print(20*'~2')
    #print(X_train.shape)

    X_train = X_train[:, :, :, None]#[:100]




    # X_train = X_train.reshape((X_train.shape, 1) + X_train.shape[1:])
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    for epoch in range(num_epoch):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            if index % 50 == 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save('../plotsDCGAN/'+
                    str(epoch)+"_"+str(index)+".png")
            
#            print(10*'~----~3~~~~')
#            print(np.shape(image_batch))
#            print(np.shape(generated_images))
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = d.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            d.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            if index % 10 == 9:
                g.save_weights('../../ModelOutDCGAN/generator', True)
                d.save_weights('../../ModelOutDCGAN/discriminator', True)


def generate(BATCH_SIZE, nice=False):
    g = generator_model()
    print(g.summary())
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('../../ModelOutDCGAN/generator')
    if nice:
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('../../ModelOutDCGAN/discriminator')
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)

        plt.figure(2)
        plt.imshow(generated_images)
        plt.savefig('../plotsDCGAN/generated_images_0.png')

    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "../plotsDCGAN/generated_image_1.png")

    plt.figure(1)
    plt.imshow(image)
    plt.savefig('../plotsDCGAN/generated_image_2.png')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
 
    num_epoch = 5


    Dir0 = '../../'
    #Dir0 = '/home/nes/Desktop/ConvNetData/lens/'
    Dir1 = Dir0 + 'AllTrainTestSets/JPG/'
    Dir2 = ['single/', 'stack/'][1]
    Dir3 = ['0/', '1/'][1]
    data_path = Dir1 + Dir2 + Dir3 + 'TrainingData/'
    names = ['lensed', 'unlensed']

    ##-------------------------------------------------------------------------------------

    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
