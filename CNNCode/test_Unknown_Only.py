## Launch training

import os, sys
import ConfigParser
import numpy as np

# Keras
import keras
from keras.callbacks import ModelCheckpoint , ReduceLROnPlateau
from keras.layers import Input, Concatenate, concatenate, Conv2D, MaxPooling2D, UpSampling2D, core, Dropout, Permute
from keras.layers import Activation, Flatten, Dense, ZeroPadding2D, AveragePooling2D, BatchNormalization , Reshape 
from keras.models import Model , Sequential, model_from_json
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras import backend as K
from keras.metrics import categorical_accuracy, binary_accuracy
from keras.losses import categorical_crossentropy , binary_crossentropy

#Custon Functions
from RPC_CNN_Functions import *

from scipy import ndimage
from PIL import Image

# import cv2
# from cv2 import CV_32F


sys.path.insert(0, './lib/')


print "before reading file"

print sys.argv
configFile = sys.argv[1]
configFilePath = './' + configFile

print configFilePath

#config file to read from
config = ConfigParser.RawConfigParser()
config.readfp(open(configFilePath))

#Load settings from Config file
config.read(configFile)
print "read configuration text"


#patch to the datasets
data_path = config.get('data paths', 'path_local')
trained_path = config.get('data paths', 'trained_path')

#Experiment name
experiment_name = config.get('experiment name', 'name')
experiment_path = './' +experiment_name +'/'

#========== Create a folder for the results
result_dir = experiment_name
print "\n1. Create directory for the results (if not already existing)"
if os.path.exists(result_dir):
    print "Dir already existing"
elif sys.platform=='win32':
    os.system('mkdir ' + result_dir)
else:
    os.system('mkdir -p ' +result_dir)

print "copy the configuration file in the results folder"
if sys.platform=='win32':
    os.system('copy ' + configFile + ' .\\' +experiment_name+'\\'+experiment_name+'_'+configFile)
else:
    os.system('cp ' + configFile + ' ./' +experiment_name+'/'+experiment_name+'_'+configFile)

print "before loading data"
data_path = config.get('data paths', 'path_local')

##### Greyscale Images
AO_test_imgs_original = data_path + config.get('data paths', 'test_imgs_original')
test_imgs_orig = load_hdf5(AO_test_imgs_original)
full_img_height = test_imgs_orig.shape[2]
full_img_width = test_imgs_orig.shape[3]

#Load the saved model
model = model_from_json(open(trained_path+'architecture.json').read())
model.load_weights(trained_path + 'best_weights.h5')
model.compile(optimizer='rmsprop', loss=categorical_crossentropy,metrics=['accuracy'])

# dimension of the patches
patch_height = model.input_shape[2]
patch_width = model.input_shape[3]
#the stride in case output with average
stride_height = int(config.get('testing settings', 'stride_height'))
stride_width = int(config.get('testing settings', 'stride_width'))
assert (stride_height < patch_height and stride_width < patch_width)

#N full images to be predicted
Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
print "number of Images"
print Imgs_to_test
#Grouping of the predicted images
N_visual = int(config.get('testing settings', 'N_group_visual'))


#============ Load the data and divide in patches
patches_imgs_test = None
new_height = None
new_width = None
border_test  = None
patches_border_test = None

patches_imgs_test, new_height, new_width = get_data_unknown_Imgs(
    test_imgs_original = data_path + config.get('data paths', 'test_imgs_original'),  #original
    imgs_to_test = int(config.get('testing settings', 'full_images_to_test')),
    patch_height = patch_height,
    patch_width = patch_width,
    stride_height = stride_height,
    stride_width = stride_width)


print "images divided into patches"



#Calculate the predictions
predictions = model.predict(patches_imgs_test, batch_size=32, verbose=2)
print "predicted images size :"
print predictions.shape

#===== Convert the prediction arrays in corresponding images

pred_background_patches, pred_capillary_patches, pred_border_patches, pred_LGVessel_patches = pred_to_multiple_imgs(predictions, patch_height, patch_width, "original")

T_background_patches, T_capillary_patches, T_border_patches, T_LGVessel_patches = pred_to_multiple_imgs(predictions, patch_height, patch_width, "threshold")



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
orig_imgs = test_imgs_orig[0:pred_imgs_capillary.shape[0],:,:,:]    #originals

#Thresholded
t_imgs_capillary = recompone_overlap(T_capillary_patches, new_height, new_width, stride_height, stride_width)# predictions
t_imgs_LGVessel = recompone_overlap(T_LGVessel_patches, new_height, new_width, stride_height, stride_width)# predictions
t_imgs_Border = recompone_overlap(T_border_patches, new_height, new_width, stride_height, stride_width)# predictions
t_imgs_background = recompone_overlap(T_background_patches, new_height, new_width, stride_height, stride_width) #predictions



## back to original dimensions
#inputs
orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]

#predictions
pred_imgs_capillary = pred_imgs_capillary[:,:,0:full_img_height,0:full_img_width]
pred_imgs_LGVessel = pred_imgs_LGVessel[:,:,0:full_img_height,0:full_img_width]
pred_imgs_Border = pred_imgs_Border[:,:,0:full_img_height,0:full_img_width]
pred_imgs_background = pred_imgs_background[:,:,0:full_img_height,0:full_img_width]

# thresholded 2
thresh2_imgs_capillary = t_imgs_capillary[:,:,0:full_img_height,0:full_img_width]
thresh2_imgs_LGVessel = t_imgs_LGVessel[:,:,0:full_img_height,0:full_img_width]
thresh2_imgs_Border = t_imgs_Border[:,:,0:full_img_height,0:full_img_width]
thresh2_imgs_background = t_imgs_background[:,:,0:full_img_height,0:full_img_width]


print "Orig imgs shape: " +str(orig_imgs.shape)
print "pred imgs shape: " +str(pred_imgs_capillary.shape)

#======== Save Images
#Save Images
visualize(group_images(orig_imgs,N_visual),experiment_path+"all_originals")#.show()


visualize(group_images(pred_imgs_capillary,N_visual),experiment_path+"all_predictions_capillary")#.show()
visualize(group_images(pred_imgs_LGVessel,N_visual),experiment_path+"all_predictions_LGVessel")#.show()
visualize(group_images(pred_imgs_Border,N_visual),experiment_path+"all_predictions_Border")#.show()
visualize(group_images(pred_imgs_background,N_visual),experiment_path+"all_predictions_Background")#.show()

visualize(group_images(thresh2_imgs_capillary,N_visual),experiment_path+"all_t2_capillary")#.show()
visualize(group_images(thresh2_imgs_LGVessel,N_visual),experiment_path+"all_t2_LGVessel")#.show()
visualize(group_images(thresh2_imgs_Border,N_visual),experiment_path+"all_t2_Border")#.show()
visualize(group_images(thresh2_imgs_background,N_visual),experiment_path+"all_t2_Background")#.show()


print "visualize over"