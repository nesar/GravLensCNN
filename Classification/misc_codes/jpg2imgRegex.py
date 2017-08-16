"""  Convert .fits files files into (45,45) .npy files.
Randomly shuffles the files and selects 20 per cent of the files for TestData.
Note: Indices of .npy and .fits are NOT the same.

    # # lsst mocks single
    # filesMocksSingleLensed0 = Dir1+'data_of_lsst/lsst_mocks_single/lensed_outputs/0/*gz'
    # filesMocksSingleLensed1 = Dir1+'data_of_lsst/lsst_mocks_single/lensed_outputs/1/*gz'
    # filesMocksSingleUnlensed0 = Dir1+'data_of_lsst/lsst_mocks_single/unlensed_outputs/0/*gz'
    # filesMocksSingleUnlensed1 = Dir1+'data_of_lsst/lsst_mocks_single/unlensed_outputs/1/*gz'
    # # lsst mocks stack
    # filesMocksStackLensed0 = Dir1+'data_of_lsst/lsst_mocks_stack/lensed_outputs/0/*gz'
    # filesMocksStackLensed1 = Dir1+'data_of_lsst/lsst_mocks_stack/lensed_outputs/1/*gz'
    # filesMocksStackUnlensed0 = Dir1+'data_of_lsst/lsst_mocks_stack/unlensed_outputs/0/*gz'
    # filesMocksStackUnlensed1 = Dir1+'data_of_lsst/lsst_mocks_stack/unlensed_outputs/1/*gz'
    # # lsst noiseless
    # filesLsstNoiselessSingle = Dir1+'lsst_noiseless_single'
    # filesLsstNoiselessStack = Dir1+'lsst_noiseless_stack'
    #
    # fileIn = filesMocksSingleLensed0
"""


import numpy as np
import matplotlib.pylab as plt
import glob
# import re



Dir1 = '/home/nes/Desktop/ConvNetData/lens/JPG/'
Dir2 = ['data_of_lsst/', 'lsst_noiseless_single/', 'lsst_noiseless_stack/'][0]
Dir3 = ['lsst_mocks_single/', 'lsst_mocks_stack/'][0]
Dir4 = ['lensed_outputs/', 'unlensed_outputs/']
Dir5 = ['0/', '1/'][1]  # 10k images each for lensed and unlensed
names = ['lensed', 'unlensed']
DirOut = '/home/nes/Desktop/ConvNetData/lens/AllTrainTestSets/JPG/'


# 'lsst_noiseless_single/', 'lsst_noiseless_stack/' have the same images as 'data_of_lsst/'
# Only lensed and noiseless - 20k images each - single and stack images

img_rows = 45
img_cols = 45
num_channel = 1
num_epoch = 2
batch_size = 8


num_classes = 2
num_files = 8000*num_classes   # Only TrainingData

num_of_samples = num_files


img_data_list=[]
labels = []
train_5para = []
test_5para = []

# for name in names:
for labelID in [0, 1]:
    name = names[labelID]
    # for img_ind in range(num_files/num_classes):
    data_path = Dir1 + Dir2 + Dir3 + Dir4[labelID] + Dir5

    print 'JPG folder: ', data_path
    fileInData = sorted(glob.glob(data_path + '*.jpg'))
    # pattern = re.compile(r"_([0-9]+)\.[^\.]+$")
    # print pattern
    # pattern.search("abc_ID_8423.pdf").group(1)
    # pattern.search(fileInData[0]).group(1)
    # fileInData




    if (len(fileInData) == 0): print 'ERROR: Empty folder'
    print 'number of files: ', len(fileInData)


    # import sys
    # sys.exit()

    np.random.seed(12345)
    alln = np.arange(len(fileInData))
    np.random.shuffle(alln)
    forTest =  alln[:int(0.2*len(fileInData))]
    # forTest = np.random.randint(10000, size = [2000])
    testind = 0
    trainind = 0

    for ind in range(len(fileInData)):
        fileIn = fileInData[ind]
        # pixel = plt.imread(fileIn)
        # if np.isnan(pixel).any(): print labelID, ind, ' -- ERROR: NaN'
        ## print ind
        ## np.save(Dir1+'TrainingData/'+Dir2+Dir3+Dir4+Dir5+str(ind), pixel)
        if ind in forTest:
            # np.save(DirOut + 'TestData/'+names[labelID]+ str(testind), pixel)
            testind+=1
            para5 = np.array(fileIn.split('_')[6:12]).astype(float)
            test_5para.append( np.hstack([labelID, para5]) )
            # print labelID
        else:
            # np.save(DirOut + 'TrainingData/'+Dir4[labelID] +names[labelID]+ str(trainind), pixel)
            trainind+=1
            para5 = np.array(fileIn.split('_')[6:12]).astype(float)
            train_5para.append( np.hstack([labelID, para5]) )
            # print (pixel).min(), (pixel).max(), names[labelID] + str(trainind)

test_5para = np.array(test_5para)
train_5para = np.array(train_5para)

def rescaled5para():
    allData = np.vstack([test_5para, train_5para])
    min5para =  np.min(allData[:,2:], axis= 0)
    max5para = np.max(allData[:,2:], axis= 0)

    print min5para
    print max5para

    test_5para[:, 2:] = (test_5para[:,2:] - min5para)/(max5para - min5para)
    train_5para[:, 2:] = (train_5para[:,2:] - min5para)/(max5para - min5para)
    # print min5para
    # print max5para

    np.save(DirOut + 'RescalingMinMax5para', np.vstack([min5para, max5para]) )
    return test_5para, train_5para

test_5para, train_5para  = rescaled5para()

    # fileIn = fileInData[2000]
    # pixel = fits.open(fileIn, memmap=True)
    #
    # plt.imshow(pixel[0].data)
    #
    # plt.show()
np.save(DirOut + 'Train5para', (train_5para) )
np.save(DirOut + 'Test5para', (test_5para) )


# print fileIn