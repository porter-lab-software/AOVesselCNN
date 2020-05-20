###################################################
#
# Script to launch the training
#
##################################################
#
# Authors: Gwen Musial, Hope Queener
# Date: 5/11/2020
#
# References:
# Musial, G., Queener, H.M., Adhikari, S., Mirhajianmoghadam, H., Schill, A.W., Patel, N.B., Porter, J.
# Automatic segmentation of retinal capillaries in adaptive optics scanning laser 
# ophthalmoscope perfusion images using a convolutional neural network
# (Pending publication)
# 
# Ronneberger O., Fischer P., Brox T. (2015) U-Net: Convolutional Networks for Biomedical Image Segmentation. 
# In: Navab N., Hornegger J., Wells W., Frangi A. (eds) 
# Medical Image Computing and Computer-Assisted Intervention MICCAI 2015
# Lecture Notes in Computer Science, vol 9351. Springer, Cham. DOI: 10.1007/978-3-319-24574-4_28
#
# Xiancheng, W. Wei, L., Bingyi, M., He, J., Jian, Z, Xu, W., Ji Z., Hong, G. (2018) 
# Retina Blood Vessel Segmentation Using A U-Net Based Convolutional Neural Network. 
# Int. Conf. Data Sci. 00, 1-11

import os, sys
import ConfigParser
import numpy as np

# Keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
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
from tensorflow.keras.initializers import VarianceScaling

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

print "before reading conguration file"

print sys.argv
configFile = sys.argv[1]
configFilePath = os.path.join('.',configFile)

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
experiment_path = os.path.join('.',experiment_name,'')
#training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))

nohup = config.getboolean('training settings', 'nohup')

## Build Model 
def get_Newnnet(n_ch,patch_height,patch_width):
    model = Sequential()
    conv2d_initializer = VarianceScaling(scale = 1.0, mode = 'fan_avg',distribution = 'uniform',seed=None)
    print '\nkernel_initializer is VarianceScaling'
#    conv2d_initializer = 'GlorotUniform'
#    print 'kernel_initalizer is GlorotUniform'
    print 'Pooling & upsampling'
    pooling_option = (4,4)
    print pooling_option
    
    batch_norm_axis = 1;
    print 'batch_norm_axis = '
    print batch_norm_axis
    

#1st Layer Section
    model.add(Conv2D(32, (25, 25), input_shape=(n_ch,patch_height,patch_width), activation='relu', padding='same',data_format='channels_first',kernel_initializer=conv2d_initializer))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (25, 25), activation='relu', padding='same',data_format='channels_first',kernel_initializer=conv2d_initializer))
    model.add(BatchNormalization(axis=batch_norm_axis, momentum=0.99, epsilon=0.001, center=True, scale=True))
    model.add(MaxPooling2D(pooling_option , data_format='channels_first'))

#2nd layer section
    model.add(Conv2D(64, (15, 15), activation='relu', padding='same',data_format='channels_first',kernel_initializer=conv2d_initializer))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (15, 15), activation='relu', padding='same',data_format='channels_first',kernel_initializer=conv2d_initializer))
    model.add(BatchNormalization(axis=batch_norm_axis, momentum=0.99, epsilon=0.001, center=True, scale=True))
    model.add(MaxPooling2D(pooling_option , data_format='channels_first'))

#3rd layer section
    model.add(Conv2D(128, (10, 10), activation='relu', padding='same',data_format='channels_first',kernel_initializer=conv2d_initializer))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (10, 10), activation='relu', padding='same',data_format='channels_first',kernel_initializer=conv2d_initializer))
    model.add(BatchNormalization(axis=batch_norm_axis, momentum=0.99, epsilon=0.001, center=True, scale=True))

#4th layer section
    model.add(UpSampling2D(size=pooling_option, data_format='channels_first'))
    model.add(Conv2D(64, (15, 15), activation='relu', padding='same',data_format='channels_first',kernel_initializer=conv2d_initializer))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (15, 15), activation='relu', padding='same',data_format='channels_first',kernel_initializer=conv2d_initializer))
    model.add(BatchNormalization(axis=batch_norm_axis, momentum=0.99, epsilon=0.001, center=True, scale=True))

#5th layer section
    model.add(UpSampling2D(size=pooling_option , data_format='channels_first'))
    model.add(Conv2D(32, (25, 25), activation='relu', padding='same',data_format='channels_first',kernel_initializer=conv2d_initializer))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (25, 25), activation='relu', padding='same',data_format='channels_first',kernel_initializer=conv2d_initializer))
    model.add(BatchNormalization(axis=batch_norm_axis, momentum=0.99, epsilon=0.001, center=True, scale=True))

#Conv6
    model.add(Conv2D(4, (1, 1), activation='relu',padding='same',data_format='channels_first',kernel_initializer=conv2d_initializer))
    model.summary()
    model.add(Reshape((4,n_ch*patch_height*patch_width)))
    model.add(Permute((2,1)))

#activation
    model.add(Activation('softmax'))

    model.compile(optimizer='rmsprop', loss=categorical_crossentropy,metrics=['categorical_accuracy'], sample_weight_mode = 'temporal')
    print 'model compile metrics: categorical_accuracy'
    print "categorical_crossentropy" 
    return model


#========== Create a folder for the results

print "\n1. Create directory for the results (if not already existing)"
if os.path.exists(experiment_path):
    print "Dir already existing"
elif sys.platform=='win32': # response in 2.7 even for win64
    os.system('mkdir ' + experiment_path)
else:
    os.system('mkdir -p ' + experiment_path)

print "copy the configuration file in the results folder"
config_record_file = experiment_name + '_' + configFile
if sys.platform=='win32': # response in 2.7 even for win64
    os.system("copy " + configFile + " " + os.path.join(experiment_path, config_record_file))
else:
    os.system("cp " + configFile + " " + os.path.join(experiment_path, config_record_file))

result_dir = experiment_name
print "before loading data"

#Load the data and divided in patches
patches_imgs_train, patches_combo_train = get_data_train_combo(
    train_imgs_original = os.path.join(data_path, config.get('data paths', 'train_imgs_original')),
    train_Combo = os.path.join(data_path, config.get('data paths','train_combo')), #4 state matrices
    patch_height = int(config.get('data attributes', 'patch_height')),
    patch_width = int(config.get('data attributes', 'patch_width')),
    stride = int(config.get('training settings', 'stride_height')))
print "loaded data"

#Save a sample of what you're feeding to the neural network 
N_sample = min(patches_imgs_train.shape[0],20)
visualize(group_images(patches_imgs_train[0:N_sample,:,:,:],5),os.path.join(experiment_path,"sample_input_imgs"))

#Convert combo image to viewable range
patches_combo_train_Vis = patches_combo_train*64
visualize(group_images(patches_combo_train_Vis[0:N_sample,:,:,:],5),os.path.join(experiment_path,"sample_input_Combo"))


#Construct and save the model arcitecture
n_ch = patches_imgs_train.shape[1]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]


model = get_Newnnet(n_ch, patch_height, patch_width)

print "\nCheck: input shape of model"
print model.input_shape

print "\nCheck: final output of the network:"
print model.output_shape

architecture_path = os.path.join(experiment_path,experiment_name +'_architecture.json');
trained_cnn = config.get('data paths','trained_cnn')
best_weights_path = os.path.join(experiment_path,trained_cnn);

# save model to .json for reference
json_string = model.to_json()
open(architecture_path, 'w').write(json_string)

#Training Parameters
# checkpointer saves best weights with model so can load entire model from single file
checkpointer = ModelCheckpoint(filepath=best_weights_path, verbose=1, monitor='loss', mode='auto', save_best_only=True) 
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
print('Done training model')
#========== Save and test the last model ===================
model.save_weights(os.path.join(experiment_path,experiment_name + '_last_weights.h5'), overwrite=True)
print "Done Saving weights"

print(model.summary())

#test the model

#========= Run the training on invariant or local
data_path = config.get('data paths', 'path_local')

#========= Run the prediction

#original test images (for FOV selection)
test_imgs_hdf5 = os.path.join(data_path,config.get('data paths', 'test_imgs_original'))
test_imgs_orig = load_hdf5(test_imgs_hdf5)
full_img_height = test_imgs_orig.shape[2]
full_img_width = test_imgs_orig.shape[3]


#Combination Matrices
test_combo_hdf5 = os.path.join(data_path, config.get('data paths', 'test_combo'))
test_combo = load_hdf5(test_combo_hdf5)

# dimension of the patches
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))
#the stride in case output with average
stride_height = int(config.get('testing settings', 'stride_height'))
stride_width = int(config.get('testing settings', 'stride_width'))
assert (stride_height < patch_height and stride_width < patch_width)

#N full images to be predicted
Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))

#Grouping of the predicted images
N_visual = int(config.get('testing settings', 'N_group_visual'))

# Load the data and divide in patches
patches_imgs_test = None
new_height = None
new_width = None

patches_imgs_test, new_height, new_width, test_imgs, test_combo = get_data_testing_overlap(
    os.path.join(data_path, config.get('data paths', 'test_imgs_original')),  #original
    os.path.join(data_path, config.get('data paths' ,'test_combo')), #combination matrix
    int(config.get('testing settings', 'full_images_to_test')),
    patch_height,
    patch_width,
    stride_height,
    stride_width)

#================ Run the prediction of the patches ==================================

#Load the saved model
model = tensorflow.keras.models.load_model(best_weights_path)
model.compile(optimizer='rmsprop', loss=categorical_crossentropy,metrics=['categorical_accuracy'])

#Calculate the predictions
predictions = model.predict(patches_imgs_test, batch_size=32, verbose=2)
print "predicted images size :"
print predictions.shape

#===== Convert the prediction arrays in corresponding images

pred_background_patches, pred_capillary_patches, pred_border_patches, pred_LGVessel_patches = pred_to_multiple_imgs(predictions, patch_height, patch_width)

#Save Predictions

# Prediction patches recombined to input image size
pred_imgs_capillary = recompone_overlap(pred_capillary_patches, new_height, new_width, stride_height, stride_width)# predictions
pred_imgs_LGVessel = recompone_overlap(pred_LGVessel_patches, new_height, new_width, stride_height, stride_width)# predictions
pred_imgs_Border = recompone_overlap(pred_border_patches, new_height, new_width, stride_height, stride_width)# predictions
pred_imgs_background = recompone_overlap(pred_background_patches, new_height, new_width, stride_height, stride_width) #predictions

#Original images
orig_imgs = None
orig_imgs = test_imgs_orig[0:pred_imgs_capillary.shape[0],:,:,:]    #originals

## back to original dimensions
#inputs
orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]
#predictions
pred_imgs_capillary = pred_imgs_capillary[:,:,0:full_img_height,0:full_img_width]
pred_imgs_LGVessel = pred_imgs_LGVessel[:,:,0:full_img_height,0:full_img_width]
pred_imgs_Border = pred_imgs_Border[:,:,0:full_img_height,0:full_img_width]
pred_imgs_background = pred_imgs_background[:,:,0:full_img_height,0:full_img_width]

print "Orig imgs shape: " +str(orig_imgs.shape)
print "pred imgs shape: " +str(pred_imgs_capillary.shape)

#Save Images
visualize(group_images(orig_imgs,N_visual),os.path.join(experiment_path,"all_originals"))#.show()
visualize(group_images(pred_imgs_capillary,N_visual),os.path.join(experiment_path,"all_predictions_capillary"))#.show()
visualize(group_images(pred_imgs_LGVessel,N_visual),os.path.join(experiment_path,"all_predictions_LGVessel"))#.show()
visualize(group_images(pred_imgs_Border,N_visual),os.path.join(experiment_path,"all_predictions_Border"))#.show()
visualize(group_images(pred_imgs_background,N_visual),os.path.join(experiment_path,"all_predictions_Background"))#.show()

print "predictions visualized"

assert (orig_imgs.shape[0]==pred_imgs_capillary.shape[0])
N_predicted = orig_imgs.shape[0]
group = N_visual
assert (N_predicted%group==0)

# These metrics are for sanity check on the code. 
# Actual preformance metrics are computed from probability images separately (not on high-performance resources)
#Compute capillary metrics based on Otsu's threshold
capillary_otsu_thresh = threshold_otsu(pred_imgs_capillary)
capillary_binary = pred_imgs_capillary > capillary_otsu_thresh
im_h = pred_imgs_capillary.shape[2]
im_w = pred_imgs_capillary.shape[3]

#Combine Predicted Images
pred_combo = combinePredictions(pred_imgs_capillary,pred_imgs_LGVessel, pred_imgs_Border,pred_imgs_background)
pred_combo = np.reshape(pred_combo,(im_h*im_w*Imgs_to_test))
test_combo = np.reshape(test_combo,(im_h*im_w*Imgs_to_test))
target_names = ['Background', 'Capillary', 'Border' , 'Large Vessel']
print(classification_report(test_combo, pred_combo, 
    target_names=target_names))

# Confusion matrix (all classes)
confusion4x4 = confusion_matrix(test_combo,pred_combo)
print "all class confusion matrix"
print confusion4x4

cap_binary_vector = np.reshape(capillary_binary,(im_h*im_w*Imgs_to_test))
cap_pred_vector = np.reshape(pred_imgs_capillary,(im_h*im_w*Imgs_to_test))
cap_true_vector = np.reshape(test_combo == 1,(im_h*im_w*Imgs_to_test))
displayPerformanceMetrics(cap_true_vector, cap_binary_vector, cap_pred_vector, experiment_path)

#Confusion matrix Capillary
print ("capillary confusion matrix")
confusion = confusion_matrix(cap_true_vector, cap_binary_vector)
displayConfusionMatrixMetrics(confusion , experiment_path)


