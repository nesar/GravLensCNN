"""
Only train over lensed images now -- so o/p also seems lensed
Have to train over everything!!!


check again if noiseless and noisy images are matching! - mostly no
Check 1d/2d issue
Convolutional encoding

"""



from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils import np_utils

from keras import backend as K

K.set_image_dim_ordering('tf')
from keras.models import load_model

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
import glob
import time

time_i = time.time()



time_i = time.time()
K.set_image_dim_ordering('tf')

from keras.preprocessing.image import ImageDataGenerator
data_augmentation = True

batch_size = 32
num_classes = 2
num_epoch = 20
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
x_test = noiseless_data[num_files:]
y_test = noiseless_target[num_files:]



# -------------------------------------------------

DirIn = '/home/nes/Dropbox/Argonne/lensData/ModelOutEncode/jupiter/ModelOutEncode/'
glob.glob(DirIn + '*autoGenerate*200*hdf5')


hyperpara = '*0.001*200*'


filelistAuto = sorted(glob.glob(DirIn + '*auto*Generate'+hyperpara + '*.hdf5'))
filelistEncode = sorted(glob.glob(DirIn + 'encodeGenerate'+hyperpara + '*.hdf5'))
filelistDecode = sorted(glob.glob(DirIn + 'decodeGenerate'+hyperpara + '*.hdf5'))
histlist = sorted(glob.glob(DirIn + '*Generate' +hyperpara + '*.npy'))

print(len(filelistAuto))

for i in range(len(filelistAuto)):
    fileInAuto = filelistAuto[i]
    fileInEncode =filelistEncode[i]
    fileInDecode = filelistDecode[i]
    histIn = histlist[i]

    loaded_model = load_model(fileInAuto)
    # print(fileIn)

    encoder = load_model(fileInEncode)
    # print(fileIn)

    decoder = load_model(fileInDecode)
    # print(fileIn)

    history = np.load(histIn)
    # print(histIn)




# fnames = data_path + DirOutType[1]  #noisy1
# noisy_data, noisy_target = load_train(fnames)
# x_test_noisy = noisy_data[num_files:]
# y_test_noisy = noisy_target[num_files:]
#
# plotCheck = False
# if plotCheck:
#
#     fig, ax = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(10, 5))
#     plt.suptitle(DirOutType[2])
#
#     count = 0
#     np.random.seed(1234)
#     indx = np.random.randint(20, size = 8)
#
#     for ind in indx:
#         pixel = x_train[ind].reshape(image_size, image_size)
#         # for i in range(numPlots):
#         ax[count / 4, count % 4].imshow(pixel, cmap=plt.get_cmap('gray'))
#         ax[count / 4, count % 4].set_title(str(ind))
#
#         count += 1

    #
    # fig, ax = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(10, 5))
    # plt.suptitle(DirOutType[1])
    #
    # count = 0
    # for ind in indx:
    #     pixel = x_train_noisy[ind].reshape(image_size, image_size)
    #     # for i in range(numPlots):
    #     ax[count / 4, count % 4].imshow(pixel, cmap=plt.get_cmap('gray'))
    #     ax[count / 4, count % 4].set_title(str(ind))
    #
    #     count += 1


#-----------------------------------------------------------------------------------

# Testing on fresh data
ifTest = True
if ifTest:

    # fileOut = 'Stack_opti' + str(opti_id) + '_loss' + str(loss_id) + '_lr' + str(
    #     learning_rate) + '_decay' + str(decay_rate) + '_batch' + str(batch_size) + '_epoch' + str(
    #     num_epoch)
    #
    #
    # loaded_model = load_model('ModelOutEncode/autoGenerate_' + fileOut + '.hdf5')
    # encoder = load_model('ModelOutEncode/encodeGenerate_' + fileOut + '.hdf5')
    # decoder = load_model('ModelOutEncode/decodeGenerate_' + fileOut + '.hdf5')



    Check_model = False
    if Check_model:
        loaded_model.summary()
        loaded_model.get_config()
        loaded_model.layers[0].get_config()
        loaded_model.layers[0].input_shape
        loaded_model.layers[0].output_shape
        loaded_model.layers[0].get_weights()
        np.shape(loaded_model.layers[0].get_weights()[0])
        loaded_model.layers[0].trainable

        from keras.utils.vis_utils import plot_model
        plot_model(loaded_model, to_file='auto_model_100runs_test.png', show_shapes=True)

    encoded_imgs = encoder.predict(x_test)

    # decoded_imgs = decoder.predict(encoded_imgs)


#
# # Totally random encoded input
    np.random.seed(2)
    GeneratedImgs = 20
#     fake_encoded_imgs = np.random.rand(GeneratedImgs*encoded_imgs[0].size).reshape(np.hstack([GeneratedImgs, encoded_imgs[0].shape] ) )

# partly random encoded input
    fake_encoded_imgs = (0.3*np.random.rand(GeneratedImgs*encoded_imgs[0].size).reshape(np.hstack([GeneratedImgs, encoded_imgs[0].shape] ) )  ) + encoded_imgs[:GeneratedImgs]

    decoded_imgs = decoder.predict(fake_encoded_imgs)




#     # decode_from_model =  loaded_model.predict(x_test_noisy)   # Same as just decoder above
#
#


#     # working with just the decoder
#
#     # np.random.seed(10)
#     # ip = np.random.rand(64)
#     # ip = np.expand_dims(ip, axis=0)
#     # # decoder.predict(ip).reshape(45, 45)
#     # plt.imshow(decoder.predict(ip).reshape(45, 45))






    plotSample = True
    if plotSample:
        n = 4
        np.random.seed(2)
        pltrange = np.random.randint(decoded_imgs.shape[0], size=n)
        fig = plt.figure(figsize=(12, 4))
        for i in range(4):
            pltID = pltrange[i]
            # display original
            ax = plt.subplot(3, n, i + 1)
            ax.set_title(str(pltID))
            plt.imshow(x_test[pltID].reshape(image_size, image_size))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display original
            # ax = plt.subplot(3, n, i + 1 + n)
            # plt.imshow(x_test_noisy[pltID].reshape(image_size, image_size))
            # plt.gray()
            # ax.get_xaxis().set_visible(False)
            # ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(3, n, i + 1 + 2*n)
            plt.imshow(decoded_imgs[pltID].reshape(image_size, image_size))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()


    plotLoss = True
    if plotLoss:
        import matplotlib.pylab as plt

        # fileOut = 'Stack_opti' + str(opti_id) + '_loss' + str(loss_id) + '_lr' + str(
        #     learning_rate) + '_decay' + str(decay_rate) + '_batch' + str(batch_size) + '_epoch' + str(
        #     num_epoch)
        #
        # history = np.load('ModelOutEncode/Generate' + fileOut+'.npy')

        epochs = history[0]
        train_loss= history[1]
        val_loss= history[2]
        # train_acc= ModelFit.history['acc']
        # val_acc= ModelFit.history['val_acc']


        fig, ax = plt.subplots(1,1, sharex= True, figsize = (7,5))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace= 0.02)
        ax.plot(epochs,train_loss)
        ax.plot(epochs,val_loss)
        ax.set_ylabel('loss')
        # ax.set_ylim([0.07,0.1])
        # ax[0].set_title('Loss')
        ax.legend(['train_loss','val_loss'])

        # accuracy doesn't make sense for regression

        plt.show()



time_j = time.time()
print(time_j - time_i, 'seconds')