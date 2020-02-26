#==========================================================
#
#  Wrapper around Prep Data Script
#
#============================================================

import os, sys
import ConfigParser
from PIL import Image
print sys.argv
import numpy as np
from RPC_CNN_Functions import *


configFile = sys.argv[1]
configFilePath = './' + configFile

#config file to read from
config = ConfigParser.RawConfigParser()
config.readfp(open(configFilePath))

#Load settings from Config file
config.read(configFile)

#path to the images
image_path = config.get('data paths', 'image_path')

#path to save hdf5 datasets
dataset_path = config.get('data paths', 'path_local')

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

#Name for HDF5 File
dataset_name = config.get('dataset name', 'name')

#Name of Image Directory
imgs_dir = image_path + 'Images/'
(os.stat(imgs_dir))

#Get information about number of images and their size
for path, subdirs, files in os.walk(imgs_dir):
    print 'Number of Images'
    Nimgs = len(files)
    print Nimgs
    img = Image.open(imgs_dir+files[0])
    img = np.asarray(img)
    print 'image dimensions'
    print img.shape
    height = img.shape[0]
    width = img.shape[1]
    if len(img.shape) > 2:
        print('Image') + files[0] + (' is RGB format, need to convert to 2 dimensions')
        raise ValueError
    for imageIndex in range(1,(Nimgs)):
        img = Image.open(imgs_dir+files[imageIndex])
        img = np.asarray(img)
        if img.shape[0] != height:
            print 'Image ' + files[imageIndex] + ' is not ' + str(height) + ' pixels high'
            raise ValueError
        if img.shape[1] != width:
            print('Image ' + files[imageIndex] + ' is not ' + str(width) + ' pixels wide')
            raise ValueError
        if len(img.shape) > 2:
             print('Image') + files[imageIndex] + (' is RGB format, need to convert to 2 dimensions')
             raise ValueError


trainingData = config.getboolean('prep data type', 'Training')
validationData = config.getboolean('prep data type', 'TestingKnown')
testingData = config.getboolean('prep data type', 'TestingUnknown')

if testingData == True:
    imgs_test = get_datasets_unknown(imgs_dir,Nimgs,height,width)
    write_hdf5(imgs_test,dataset_path + dataset_name + "_imgs_test.hdf5")

if trainingData ==True:
    imgs_dir_train = imgs_dir
    groundTruth_dir_train = image_path + 'manual/'
    border_dir_train = image_path + 'border/'
    LGV_dir_train = image_path + 'LGVessels/'
    combo_dir_train = image_path + 'combined/'
    imgs_train, groundTruth_train, border_masks_train , LGvessel_masks_train, Combos_train = get_datasets(imgs_dir_train,groundTruth_dir_train,border_dir_train, LGV_dir_train, combo_dir_train ,Nimgs , height, width, "train")
    
    print("saving training datasets")
    write_hdf5(imgs_train, dataset_path + dataset_name + "_imgs_train.hdf5")
    write_hdf5(groundTruth_train, dataset_path + dataset_name + "_groundTruth_train.hdf5")
    write_hdf5(border_masks_train,dataset_path + dataset_name+ "_borderMasks_train.hdf5")
    write_hdf5(LGvessel_masks_train,dataset_path + dataset_name+ "_LGVesselMasks_train.hdf5")
    write_hdf5(Combos_train,dataset_path + dataset_name+ "_Combos_train.hdf5")

if validationData == True: 
    val_image_path = config.get('data paths', 'validation_image_path')
    val_imgs_dir = val_image_path + 'Images/'
    print val_imgs_dir
    for path, subdirs, files in os.walk(val_imgs_dir):
        Nimgs_val = len(files)
        img = Image.open(val_imgs_dir+files[0])
        img = np.asarray(img)
        height = img.shape[1]
        width = img.shape[0]
    
    groundTruth_dir_val = val_image_path + 'manual/'
    border_dir_val = val_image_path + 'border/'
    LGV_dir_val = val_image_path + 'LGVessels/'
    combo_dir_val = val_image_path + 'combined/'
    imgs_test, groundTruth_test, border_masks_test , LGvessel_masks_test ,Combos_test = get_datasets(val_imgs_dir,groundTruth_dir_val,border_dir_val, LGV_dir_val, combo_dir_val ,Nimgs_val , height, width, "test")
    
    print("saving validation datasets")
    write_hdf5(imgs_test,dataset_path + dataset_name + "_imgs_test.hdf5")
    write_hdf5(groundTruth_test, dataset_path + dataset_name + "_groundTruth_test.hdf5")
    write_hdf5(border_masks_test,dataset_path + dataset_name + "_borderMasks_test.hdf5")
    write_hdf5(LGvessel_masks_test,dataset_path + dataset_name +"_LGVesselMasks_test.hdf5")
    write_hdf5(Combos_test,dataset_path + dataset_name +"_Combos_test.hdf5")
