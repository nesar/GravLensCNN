#!/usr/bin/python
'''
ConvNet implementation of classification
Runs with default parameters.

Dataset:
    Datapath for Training data must have 2 directories - labeled_outputs and unlabeled_outputs
    Each file are named as labeled<ID>.npy or unlabeled<ID>.npy

    can be generated from jpg2imgRegex.py

'''

import numpy as np
import time
from keras.preprocessing.image import ImageDataGenerator
from model_architectures import basic_model
import load_train_data


def train(model, X_train, y_train, num_epoch = 200, batch_size = 32, train_val_split = 0.2,
          DataAugmentation = True):


    time_i = time.time()

    if DataAugmentation:

        print('Implementing pre-process and (real-time) data-augmentation (Check default options)')

        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range= 0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range= 0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True)  # randomly flip images

        datagen.fit(X_train)

        # Fit the model on the batches.
        ModelFit = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                            steps_per_epoch=X_train.shape[0] // batch_size,
                            epochs=num_epoch,
                            validation_data= (X_test, y_test ), verbose=2)

    else:
        print('No pre-processing data-augmentation')
        ModelFit = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=num_epoch,
                             verbose=1, validation_split= (1.0 - train_val_split) )
        # ModelFit = model.fit(X_train, y_train, batch_size= batch_size, nb_epoch=num_epoch, verbose=1, validation_data=(X_test, y_test))


        time_j = time.time()
        print('Training time:  ', time_j - time_i, 'seconds')

    return ModelFit


def saveModel(ModelFit, fileOut):


    train_loss = ModelFit.history['loss']
    val_loss = ModelFit.history['val_loss']
    train_acc = ModelFit.history['acc']
    val_acc = ModelFit.history['val_acc']

    epochs = np.arange(1, np.size(train_loss)+1)

    training_hist = np.vstack([epochs, train_loss, val_loss, train_acc, val_acc])

    # fileOut =

    model.save(fileOut+'.hdf5')
    np.save(fileOut+'.npy', training_hist)

    print('final acc - train and val')
    print(train_acc[-1], val_acc[-1])

if __name__ == "__main__":


    num_epoch = 10
    batch_size = 8
    learning_rate = 0.01  # Warning: lr and decay vary across optimizers
    decay_rate = 0.0
    opti_id = 0  # [SGD, Adam, RMSprop]
    loss_id = 0

    # image_size = 45
    # num_channel = 1
    # num_classes = 2
    # num_files = 8000*num_classes
    # train_split = 0.8   # 80 percent
    # num_train = int(train_split*num_files)


    # Dir0 = '../../'
    Dir0 = '/home/nes/Desktop/ConvNetData/lens/'
    Dir1 = Dir0 + 'AllTrainTestSets/JPG/'
    Dir2 = ['single/', 'stack/'][1]
    Dir3 = ['0/', '1/'][1]
    data_path = Dir1 + Dir2 + Dir3 + 'TrainingData/'
    names = ['lensed', 'unlensed']
    data_dir_list = ['lensed_outputs', 'unlensed_outputs']


    ##-------------------------------------------------------------------------------------
    ## Load data

    lens = load_data.LensData(data_path = Dir1 + Dir2 + Dir3 + 'TrainingData/')
    (X_train, y_train), (X_test, y_test) = lens.load_data()[:100]

    ##-------------------------------------------------------------------------------------
    ## Create model

    # model =  create_model()  # Default network
    #model = create_model_deeper()  # Deeper ConvNet
    #model = create_model(learning_rate = learning_rate, decay_rate = decay_rate, opti_id = opti_id, loss_id = loss_id)   # Custom parameters
 
    model = basic_model()
    print (model.summary())

    ##-------------------------------------------------------------------------------------
    ## Fit model

    # ModelFit = train(model, X_train, y_train)  # Train with default values
    ModelFit = train(model, X_train, y_train, num_epoch=num_epoch, batch_size=batch_size,
                     DataAugmentation = True)  ## Train with customized parameters

    ##-------------------------------------------------------------------------------------
    ## Save model and history
    DirOut = './ModelOutClassification/'

    fileOut = DirOut + 'LensJPG_stack_opti' + str(opti_id) + '_loss' + str(loss_id) + '_lr' + str(learning_rate) + '_decay' + str(decay_rate) + '_batch' + str(batch_size) + '_epoch' + str(num_epoch)

    saveModel(ModelFit, fileOut)

    #--------------------------------------------------------------------------------------------------



