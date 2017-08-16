import numpy as np
from keras.utils import np_utils

from keras import backend as K
K.set_image_dim_ordering('tf')


class LensData:
    '''
        Usage - lensData().load_data

        1) input_images: loads npy files, - shuffled randomly.
        2) normalize_data: re-scaling (-1, 1)
        3) load_data: returns 2 tuples: (x_train, y_train), (x_test, y_test)
            where y_train and y_test are already one-hot-encoded (using keras.np_utils)
    '''
    def __init__(self, num_classes = 2, num_channel = 1, train_val_split = 0.8, files_per_class =
    8000, data_path = './', names=['lensed', 'unlensed']):

        self.num_channel = num_channel
        self.num_classes = num_classes
        self.files_per_class = files_per_class
        self.num_files = self.files_per_class*self.num_classes
        self.train_val_split = train_val_split
        self.num_train = int(self.train_val_split*self.num_files)

        self.names = names
        self.data_path = data_path


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
                name = self.names[labelID]

                # print(labelID)
                input_img = np.load(self.data_path + '/' + name + '_outputs/' + name + str(img_ind) + '.npy')
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
        print (img_data.shape)

        if self.num_channel == 1:
            if K.image_dim_ordering() == 'th':
                img_data = np.expand_dims(img_data, axis=1)
                print (img_data.shape)
            else:
                img_data = np.expand_dims(img_data, axis=4)
                print (img_data.shape)
        else:
            if K.image_dim_ordering() == 'th':
                img_data = np.rollaxis(img_data, 3, 1)
        print (img_data.shape)


        self.train_data = img_data
        # labels = np.load(Dir1 + Dir2 + Dir3 + 'Train5para.npy')
        print (labels)

        self.train_target = np_utils.to_categorical(labels, self.num_classes)

        # print(self.train_target)

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
        print('Train shape:', train_data.shape)
        print(train_data.shape[0], 'train samples')
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


