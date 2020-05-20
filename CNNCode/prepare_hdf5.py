###################################################
#
# Script to concatonate data for training as hdf5
#
###################################################
#
# Authors: Gwen Musial, Hope Queener
# Date: 5/11/2020
#
# Converts individual image files to continguous hdf5 
# for high-performance computing resources
# Folders are expected to contain images of uniform size
# that are concatonated into one large mulitdimensional array
# No other data should be in each folder
#

import os, sys
import ConfigParser
from PIL import Image
print sys.argv
import numpy as np
from RPC_CNN_Functions import *


configFile = sys.argv[1]
configFilePath = os.path.join('.',configFile)

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

#Image and combo ground truth files are expected to be in these subfolders
images_subfolder = 'Images'
combined_subfolder = 'combined'
imgs_dir = os.path.join(image_path,images_subfolder,'')
(os.stat(imgs_dir))

trainingData = config.getboolean('prep data type', 'Training')
validationData = config.getboolean('prep data type', 'TestingKnown')

#Get information about number of images and their size
for path, subdirs, files in os.walk(imgs_dir):
    print 'Number of Images'
    files.sort()
    Nimgs = len(files)
    print Nimgs
    pil_img = Image.open(os.path.join(imgs_dir,files[0]))
    img = np.asarray(pil_img)
    pil_img.close()    
    print 'Image dimensions'
    print img.shape
    height = img.shape[0]
    width = img.shape[1]
    if len(img.shape) > 2:
        print('Image') + files[0] + (' is RGB format. Image must be converted to 2 dimensions (grayscale).')
        raise ValueError
    for imageIndex in range(1,(Nimgs)):
        pil_img = Image.open(os.path.join(imgs_dir,files[imageIndex]))
        img = np.asarray(pil_img)
        pil_img.close()        
        if len(img.shape) > 2:
             print(('Image') + files[imageIndex] + (' is RGB format. Image must be converted to 2 dimensions (grayscale).'))
             raise ValueError
        if img.shape[0] != height:
            print 'Image ' + files[imageIndex] + ' is not ' + str(height) + ' pixels high.'
            raise ValueError
        if img.shape[1] != width:
            print('Image ' + files[imageIndex] + ' is not ' + str(width) + ' pixels wide.')
            raise ValueError


if trainingData ==True:
    imgs_dir_train = imgs_dir
    train_imgs_original = config.get('data paths','train_imgs_original')
    combo_dir_train = os.path.join(image_path , combined_subfolder,'')
    train_combo = config.get('data paths','train_combo')
    imgs_train, Combos_train = get_datasets(imgs_dir_train, combo_dir_train, Nimgs , height, width)
    print imgs_train.shape
    imgs_train_hdf5 = os.path.join(dataset_path, train_imgs_original)
    print(imgs_train_hdf5)
    write_hdf5(imgs_train,imgs_train_hdf5 )
    write_hdf5(Combos_train,os.path.join(dataset_path, train_combo))
    
    print("Training datasets saved as HDF5.")

if validationData == True: 
    val_image_path = config.get('data paths', 'validation_image_path')
    test_imgs_original = config.get('data paths','test_imgs_original')
    test_combo = config.get('data paths','test_combo')
    val_imgs_dir = os.path.join(val_image_path,images_subfolder,'');
    print val_imgs_dir
    for path, subdirs, files in os.walk(val_imgs_dir):
        files.sort()
        Nimgs_val = len(files)
        pil_img = Image.open(os.path.join(val_imgs_dir,files[0]))
        img = np.asarray(pil_img)
        pil_img.close()
        height = img.shape[1]
        width = img.shape[0]
    
    combo_dir_val = os.path.join(val_image_path,combined_subfolder,'')
    imgs_test, Combos_test = get_datasets(val_imgs_dir,combo_dir_val ,Nimgs_val , height, width)
    
    write_hdf5(imgs_test,os.path.join(dataset_path, test_imgs_original))
    write_hdf5(Combos_test,os.path.join(dataset_path, test_combo))
    print("Validation datasets saved as HDF5.")
