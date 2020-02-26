###################################################
#
#   Script to launch the training
#
##################################################

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

#import cv2
#from cv2 import CV_32F

# scikit learn
from sklearn.metrics import confusion_matrix , f1_score, jaccard_similarity_score , roc_curve
from sklearn.metrics import precision_recall_curve, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score

from skimage.filters import threshold_otsu

sys.path.insert(0, './lib/')

print "before reading file"

print sys.argv
configFile = sys.argv[1]
configFilePath = './' + configFile

#config file to read from
config = ConfigParser.RawConfigParser()
config.readfp(open(configFilePath))

#Load settings from Config file
config.read(configFile)
print "read configuration text"
#path to the datasets
data_path = config.get('data paths', 'path_local')
#Experiment name
experiment_name = config.get('experiment name', 'name')
experiment_path = './' +experiment_name +'/'
#training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))

nohup = config.getboolean('training settings', 'nohup')

## Build Model 
def get_Newnnet(n_ch,patch_height,patch_width):
    model = Sequential()

#1st Layer Section
    model.add(Conv2D(32, (25, 25), input_shape=(n_ch,patch_height,patch_width), activation='relu', padding='same',data_format='channels_first'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (25, 25), activation='relu', padding='same',data_format='channels_first'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
    model.add(MaxPooling2D((2, 2) , data_format='channels_first'))

#2nd layer section
    model.add(Conv2D(64, (15, 15), activation='relu', padding='same',data_format='channels_first'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (15, 15), activation='relu', padding='same',data_format='channels_first'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
    model.add(MaxPooling2D((2, 2) , data_format='channels_first'))

#3rd layer section
    model.add(Conv2D(128, (10, 10), activation='relu', padding='same',data_format='channels_first'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (10, 10), activation='relu', padding='same',data_format='channels_first'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))

#4th layer section
    model.add(UpSampling2D(size=(2, 2), data_format='channels_first'))
    model.add(Conv2D(64, (15, 15), activation='relu', padding='same',data_format='channels_first'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (15, 15), activation='relu', padding='same',data_format='channels_first'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))

#5th layer section
    model.add(UpSampling2D(size=(2, 2) , data_format='channels_first'))
    model.add(Conv2D(32, (25, 25), activation='relu', padding='same',data_format='channels_first'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (25, 25), activation='relu', padding='same',data_format='channels_first'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))

#Conv6
    model.add(Conv2D(4, (1, 1), activation='relu',padding='same',data_format='channels_first'))
    model.summary()
    model.add(Reshape((4,n_ch*patch_height*patch_width)))
    model.add(Permute((2,1)))

#activation
    model.add(Activation('softmax'))

    model.compile(optimizer='rmsprop', loss=categorical_crossentropy,metrics=['accuracy'], sample_weight_mode = 'temporal')
    return model

print "axis = -1"
print "categorical_crossentropy" 
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

#Load the data and divided in patches
patches_imgs_train, patches_borders_train ,patches_combo_train = get_data_train_combo(
    train_imgs_original = data_path + config.get('data paths', 'train_imgs_original'),
    train_border = data_path + config.get('data paths','train_border_masks'),#mask
    train_Combo = data_path + config.get('data paths','train_combo'), #4 state matrices
    patch_height = int(config.get('data attributes', 'patch_height')),
    patch_width = int(config.get('data attributes', 'patch_width')),
    stride = int(config.get('training settings', 'stride_height')))
print "loaded data"

#Save a sample of what you're feeding to the neural network 
N_sample = min(patches_imgs_train.shape[0],20)
visualize(group_images(patches_imgs_train[0:N_sample,:,:,:],5),'./'+experiment_name+'/'+"sample_input_imgs")
visualize(group_images(patches_borders_train[0:N_sample,:,:,:],5),'./'+experiment_name+'/'+"sample_input_border")

#Convert combo image to viewable range
patches_combo_train_Vis = patches_combo_train*64
visualize(group_images(patches_combo_train_Vis[0:N_sample,:,:,:],5),'./'+experiment_name+'/'+"sample_input_Combo")


#Construct and save the model arcitecture
n_ch = patches_imgs_train.shape[1]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]


model = get_Newnnet(n_ch, patch_height, patch_width)

print "\nCheck: input shape of model"
print model.input_shape

print "\nCheck: final output of the network:"
print model.output_shape

json_string = model.to_json()
open('./'+experiment_name+'/'+experiment_name +'_architecture.json', 'w').write(json_string)

#Training Parameters
checkpointer = ModelCheckpoint(filepath='./'+experiment_name+'/'+experiment_name +'_best_weights.h5', verbose=1, monitor='loss', mode='auto', save_best_only=True) 
reduction = ReduceLROnPlateau(monitor='loss', factor=0.02, patience=5, verbose=1, mode='auto', min_delta=0.001, cooldown=0, min_lr=0.0000001)

#Print Number of Patches
print "\nNumber of Patches"
print patches_combo_train.shape[0]

#Add weights
reShapedWeights , kerasLabeledPatchesArray = addWeightsToPatches(patches_combo_train)


#Prints the percent of each class in the training set
combinationComposition(patches_combo_train,kerasLabeledPatchesArray)

print "\nshape of weights"
print reShapedWeights.shape


model.fit(patches_imgs_train, kerasLabeledPatchesArray, epochs=N_epochs, batch_size=batch_size, sample_weight = reShapedWeights , verbose=2, shuffle=True, validation_split=0.25, callbacks=[checkpointer, reduction])

#========== Save and test the last model ===================
model.save_weights('./'+experiment_name+'/'+experiment_name +'_last_weights.h5', overwrite=True)


print "Done Training"
print(model.summary())

#test the model

#========= Run the training on invariant or local
data_path = config.get('data paths', 'path_local')

#========= Run the prediction

#original test images (for FOV selection)
AO_test_imgs_original = data_path + config.get('data paths', 'test_imgs_original')
test_imgs_orig = load_hdf5(AO_test_imgs_original)
full_img_height = test_imgs_orig.shape[2]
full_img_width = test_imgs_orig.shape[3]
#the border masks
AO_test_border_masks = data_path + config.get('data paths', 'test_border_masks')
test_border_masks = load_hdf5(AO_test_border_masks)

#The Capillary ground truth
AO_test_Capillaries = data_path + config.get('data paths', 'test_groundTruth')
test_Capillaries = load_hdf5(AO_test_Capillaries)

#The Large vessel ground truth
AO_test_LGVessels = data_path + config.get('data paths', 'test_LGVessels')
test_LGVessels = load_hdf5(AO_test_LGVessels)


#Combination Matrices
AO_test_Combo = data_path + config.get('data paths', 'test_combo')
test_combo = load_hdf5(AO_test_Combo)

# dimension of the patches
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))
#the stride in case output with average
stride_height = int(config.get('testing settings', 'stride_height'))
stride_width = int(config.get('testing settings', 'stride_width'))
assert (stride_height < patch_height and stride_width < patch_width)
#model name
experiment_name = config.get('experiment name', 'name')
experiment_path = './' +experiment_name +'/'
#N full images to be predicted
Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))

#Grouping of the predicted images
N_visual = int(config.get('testing settings', 'N_group_visual'))

# Load the data and divide in patches
patches_imgs_test = None
new_height = None
new_width = None
border_test  = None
patches_border_test = None
patches_imgs_test, new_height, new_width, test_Capillaries, border_test, test_combo, test_LGVessels , patches_capillary_test , patches_border_test , patches_LGV_test, patches_combo_test = get_data_testing_overlap(
    test_imgs_original = data_path + config.get('data paths', 'test_imgs_original'),  #original
    test_Capillaries = data_path + config.get('data paths', 'test_groundTruth'),  #capillaries
    test_border = data_path + config.get('data paths','test_border_masks'),  #borders
    test_combo = data_path + config.get('data paths' ,'test_combo'), #combination matrix
    test_LGVessels = data_path + config.get('data paths' , 'test_LGVessels'), #LG Vessels
    imgs_to_test = int(config.get('testing settings', 'full_images_to_test')),
    patch_height = patch_height,
    patch_width = patch_width,
    stride_height = stride_height,

    stride_width = stride_width)


#================ Run the prediction of the patches ==================================

#Load the saved model

model = model_from_json(open(experiment_path+experiment_name +'_architecture.json').read())
model.load_weights(experiment_path+experiment_name + '_best_weights.h5')
model.compile(optimizer='rmsprop', loss=categorical_crossentropy,metrics=['accuracy'])

#Calculate the predictions
predictions = model.predict(patches_imgs_test, batch_size=32, verbose=2)
print "predicted images size :"
print predictions.shape

#===== Convert the prediction arrays in corresponding images

pred_background_patches, pred_capillary_patches, pred_border_patches, pred_LGVessel_patches = pred_to_multiple_imgs(predictions, patch_height, patch_width, "original")

#Save Predictions

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
capillary_masks = test_Capillaries[0:pred_imgs_capillary.shape[0],:,:,:]  #ground truth masks Capillaries
lgVessel_masks = test_LGVessels[0:pred_imgs_capillary.shape[0],:,:,:] #ground truth masks LG vessels
border_masks = border_test[0:pred_imgs_capillary.shape[0],:,:,:] #ground Truth for border
combo_masks= test_combo[0:pred_imgs_capillary.shape[0],:,:,:] 


#Thresholded images
thresh_imgs_capillary  = otsuThreshold(pred_imgs_capillary)
thresh_imgs_LGVessel   = otsuThreshold(pred_imgs_LGVessel)

## back to original dimensions
#inputs
orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]
capillary_masks = capillary_masks[:,:,0:full_img_height,0:full_img_width]
lgVessel_masks = lgVessel_masks[:,:,0:full_img_height,0:full_img_width]
border_masks = border_masks[:,:,0:full_img_height,0:full_img_width]

#predictions
pred_imgs_capillary = pred_imgs_capillary[:,:,0:full_img_height,0:full_img_width]
pred_imgs_LGVessel = pred_imgs_LGVessel[:,:,0:full_img_height,0:full_img_width]
pred_imgs_Border = pred_imgs_Border[:,:,0:full_img_height,0:full_img_width]
pred_imgs_background = pred_imgs_background[:,:,0:full_img_height,0:full_img_width]

#thresholded
thresh_imgs_capillary = thresh_imgs_capillary[:,:,0:full_img_height,0:full_img_width]
thresh_imgs_LGVessel = thresh_imgs_LGVessel[:,:,0:full_img_height,0:full_img_width]


print "Orig imgs shape: " +str(orig_imgs.shape)
print "pred imgs shape: " +str(pred_imgs_capillary.shape)


#Save Images
visualize(group_images(orig_imgs,N_visual),experiment_path+"all_originals")#.show()
visualize(group_images(capillary_masks,N_visual),experiment_path+"all_capillary_Input")#.show()
visualize(group_images(lgVessel_masks,N_visual),experiment_path+"all_LGVessel_Input")#.show()
visualize(group_images(border_masks,N_visual),experiment_path+"all_Border_Input")#.show()

visualize(group_images(pred_imgs_capillary,N_visual),experiment_path+"all_predictions_capillary")#.show()
visualize(group_images(pred_imgs_LGVessel,N_visual),experiment_path+"all_predictions_LGVessel")#.show()
visualize(group_images(pred_imgs_Border,N_visual),experiment_path+"all_predictions_Border")#.show()
visualize(group_images(pred_imgs_background,N_visual),experiment_path+"all_predictions_Background")#.show()


visualize(group_images(thresh_imgs_capillary,N_visual),experiment_path+"all_t_capillary")#.show()
visualize(group_images(thresh_imgs_LGVessel,N_visual),experiment_path+"all_t_LGVessel")#.show()

print "visualize over"

assert (orig_imgs.shape[0]==pred_imgs_capillary.shape[0] and orig_imgs.shape[0]==capillary_masks.shape[0])
N_predicted = orig_imgs.shape[0]
group = N_visual
assert (N_predicted%group==0)

#Compute Metrics


cap_prob = pred_imgs_capillary
cap_true = capillary_masks
cap_thresh = thresh_imgs_capillary

im_h = pred_imgs_capillary.shape[2]
im_w = pred_imgs_capillary.shape[3]

cap_true = np.reshape(cap_true,(im_h*im_w*Imgs_to_test))
cap_thresh = np.reshape(cap_thresh,(im_h*im_w*Imgs_to_test))
cap_prob = np.reshape(cap_prob,(im_h*im_w*Imgs_to_test))


#Combine Predicted Images
pred_Combined = combinePredictions(pred_imgs_capillary,pred_imgs_LGVessel, pred_imgs_Border,pred_imgs_background)
target_names = ['Background', 'Capillary', 'Border' , 'Large Vessel']

#Save the 4 category predicion images
#predCombinedScaled = pred_Combined *66 
#visualize(group_images(predCombinedScaled,N_visual),experiment_path+"all_combined")#.show()


pred_Combined = np.reshape(pred_Combined,(im_h*im_w*Imgs_to_test))
test_combo = np.reshape(test_combo,(im_h*im_w*Imgs_to_test))


print(classification_report(test_combo, pred_Combined, target_names=target_names))

(cap_thresh !=0).astype(int)
cap_true = cap_true>0
cap_true = cap_true.astype(np.int)
(cap_true !=0) == 1
cap_thresh = cap_thresh >0
cap_thresh = cap_thresh.astype(np.int)

# Confusion matrix All Class
confusion_3 = confusion_matrix(test_combo,pred_Combined)
print "all class confusion matrix"
print confusion_3
#Confusion matrix Capillary
confusion = confusion_matrix(cap_true, cap_thresh)
displayPerformanceMetrics(cap_true, cap_thresh, cap_prob , experiment_path)
displayConfusionMatrixMetrics(confusion , experiment_path)


