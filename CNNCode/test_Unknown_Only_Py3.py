###################################################
#
# Apply prior training to images to segment 4 classes
#
###################################################
#
# Authors: Gwen Musial, Hope Queener
# Date: 6/30/2020

import os, sys
import configparser
import numpy as np
import datetime

# Keras
import tensorflow
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint , ReduceLROnPlateau
from tensorflow.keras.layers import Input, Concatenate, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Permute
#from tensorflow.keras.layers import core # removed when rcdc.uh.edu updated keras 2.2.4 to import from tensorflow 2.2.0
from tensorflow.keras.layers import Activation, Flatten, Dense, ZeroPadding2D, AveragePooling2D, BatchNormalization , Reshape 
from tensorflow.keras.models import Model , Sequential, model_from_json
from tensorflow.keras.optimizers import SGD
#from tensorflow.keras.utils import np_utils # np_utils is not part of public API (github tensorflow issue 14008)
#from tensorflow.python.keras._impl.keras.utils import np_utils
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import categorical_accuracy, binary_accuracy
from tensorflow.keras.losses import categorical_crossentropy , binary_crossentropy

#Custom Functions
from RPC_CNN_Functions import *

from scipy import ndimage
from PIL import Image

if sys.platform=='win32':
    print('Windows (32/64-bit) detected. This project was tested only on Linux system.')
    exit();
print(datetime.datetime.now())    

sys.path.insert(0, os.path.join('.','lib'))
print("before reading file")

print(sys.argv)
configFile = sys.argv[1]
configFilePath = os.path.join('.' ,configFile)

print(configFilePath)

#config file to read from
config = configparser.RawConfigParser()
config.readfp(open(configFilePath))

#Load settings from Config file
config.read(configFile)
print("read configuration text")

#patch to the datasets
image_path = config.get('data paths', 'image_path')
#data_path = config.get('data paths', 'path_local')
trained_path = config.get('data paths', 'trained_path')
trained_cnn = config.get('data paths','trained_cnn')

#Experiment name
experiment_name = config.get('experiment name', 'name')
experiment_path = os.path.join('.',experiment_name,'')

#Loop through all testing TIF images, generate the HDF5 files and run the CNN on them one-by-one-by-one
#if not os.path.exists(data_path):
#    os.makedirs(data_path)

#Name of Image Directory
imgs_dir = os.path.join(image_path,'Images')
#(os.stat(imgs_dir))
print(imgs_dir)


if os.path.exists(experiment_path):
    print("Dir already existing")
else:
    os.system('mkdir -p ' + experiment_path)
print("copy the configuration file in the results folder")
os.system('cp ' + configFile + ' ' + os.path.join(experiment_path,experiment_name+'_'+configFile))

#Load the saved model
model = tensorflow.keras.models.load_model(os.path.join(trained_path,trained_cnn))
model.compile(optimizer='rmsprop', loss=categorical_crossentropy,metrics=['categorical_accuracy'])
print("model loaded")

# dimension of the patches
patch_height = model.input_shape[2]
patch_width = model.input_shape[3]

#the stride in case output with average
stride_height = int(config.get('testing settings', 'stride_height'))
stride_width = int(config.get('testing settings', 'stride_width'))
assert (stride_height < patch_height and stride_width < patch_width)

# Process one image at a time, skipping RGB TIFs

for parent_folder, subfolders, files in os.walk(imgs_dir):
    if not files:
        print(('The folder ' + parent_folder + ' contains no files.'))
    else:
        print('The folder ' + parent_folder + ' contains ' + str(len(files)) + ' files.')        
        
        files.sort()
        for file_name in files:
            pil_img = Image.open(os.path.join(parent_folder,file_name))

            img = np.asarray(pil_img)
            pil_img.close()
            del pil_img
            
            print('image dimensions')
            print(img.shape)

            if len(img.shape) > 2:
                print(('Image ' + file_name + ' is RGB format, need to convert to 2 dimensions'))
            else:
                print(('Segmenting ' + file_name + '...'))

                #========== Create a folder for the results
                file_partitions = file_name.rpartition('.')
                if len(file_partitions[0])>0:
                    prefix = file_partitions[0]
                else:
                    prefix = 'unknown'
                
                height = img.shape[0]
                width = img.shape[1]    
                print(("imgs max: " +str(np.max(img))))
                print(("imgs min: " +str(np.min(img))))
                imgs_test = np.empty((1,height,width)) # keep format of nxhxw with n =1
                imgs_test[0] = img;   
                test_imgs_orig = np.reshape(imgs_test,(1,1,height,width))
                print(( 'test_imgs_orig dims:' + str(len(test_imgs_orig.shape))))
                print(('shape 0 = ', str(test_imgs_orig.shape)))
                
                full_img_height = test_imgs_orig.shape[2]
                full_img_width = test_imgs_orig.shape[3]

                #============ Load the data and divide in patches
                patches_imgs_test = None
                new_height = None
                new_width = None
                border_test  = None
                patches_border_test = None
                patches_imgs_test, new_height, new_width = get_data_unknown_Imgs(
                    test_imgs_orig,  #original
                    patch_height = patch_height,
                    patch_width = patch_width,
                    stride_height = stride_height,
                    stride_width = stride_width)
                print("images divided into patches")


                #Calculate the predictions
                predictions = model.predict(patches_imgs_test, batch_size=32, verbose=2)
                print("predicted images size :")
                print(predictions.shape)

                #===== Convert the prediction arrays in corresponding images
                pred_background_patches, pred_capillary_patches, pred_border_patches, pred_LGVessel_patches = pred_to_multiple_imgs(predictions, patch_height, patch_width)
                #T_background_patches, T_capillary_patches, T_border_patches, T_LGVessel_patches = pred_to_multiple_imgs(predictions, patch_height, patch_width, "threshold")

                #========== Elaborate and visualize the predicted images ====================
                pred_imgs = None
                orig_imgs = None
                capillary_masks = None

                #Keeping prediction values
                pred_imgs_capillary = recompone_overlap(pred_capillary_patches, new_height, new_width, stride_height, stride_width)# predictions
                pred_imgs_LGVessel = recompone_overlap(pred_LGVessel_patches, new_height, new_width, stride_height, stride_width)# predictions
                pred_imgs_Border = recompone_overlap(pred_border_patches, new_height, new_width, stride_height, stride_width)# predictions
                pred_imgs_background = recompone_overlap(pred_background_patches, new_height, new_width, stride_height, stride_width) #predictions

                #Original images
                orig_imgs = test_imgs_orig[0:pred_imgs_capillary.shape[0],:,:,:]    #original

                ## back to original dimensions
                #inputs
                orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]

                #predictions
                pred_imgs_capillary = pred_imgs_capillary[:,:,0:full_img_height,0:full_img_width]
                pred_imgs_LGVessel = pred_imgs_LGVessel[:,:,0:full_img_height,0:full_img_width]
                pred_imgs_Border = pred_imgs_Border[:,:,0:full_img_height,0:full_img_width]
                pred_imgs_background = pred_imgs_background[:,:,0:full_img_height,0:full_img_width]

                # thresholded 2

                print("Orig imgs shape: " +str(orig_imgs.shape))
                print("pred imgs shape: " +str(pred_imgs_capillary.shape))

                #======== Save Images
                #Save Images
                visualize(group_images(orig_imgs,1),os.path.join(experiment_path ,prefix+"_originals")) #show()

                visualize(group_images(pred_imgs_capillary,1),os.path.join(experiment_path,prefix+"_predictions_capillary")) # .show()
                visualize(group_images(pred_imgs_LGVessel,1),os.path.join(experiment_path,prefix+"_predictions_LGVessel")) # .show()
                visualize(group_images(pred_imgs_Border,1),os.path.join(experiment_path,prefix+"_predictions_canvas")) # .show()
                visualize(group_images(pred_imgs_background,1),os.path.join(experiment_path,prefix+"_predictions_background")) # .show()
print("segmentation completed")
print(datetime.datetime.now())