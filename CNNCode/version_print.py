#Prints what version of each module is being called at this time

import os
#print "os version"
#print os.__version__

import sys
print "Python sys version"
print sys.version_info

import ConfigParser
#print "ConfigParser Version"
#print ConfigParser.__version__

import h5py
print "h5py version"
print h5py.__version__

import numpy as np
print "np version"
print np.__version__

from PIL import Image
print "Image version"
print Image.__version__

import tensorflow as tf
print "TensorFlow version"
print tf.__version__

from tensorflow import keras
print "Keras Version"
print keras.__version__

import scipy
print "scipy Version"
print scipy.__version__

import skimage 
print "skimage Version"
print skimage.__version__

import sklearn.metrics
print "sklearn Version"
print sklearn.__version__

# commented out in other py modules
#import cv2
#print "cv2 Version"
#print cv2.__version__


# Keras
# test import of specific modules
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

#Custon Functions
from RPC_CNN_Functions import *

from scipy import ndimage

#import cv2
#from cv2 import CV_32F

# scikit learn
from sklearn.metrics import confusion_matrix , f1_score, jaccard_similarity_score , roc_curve
from sklearn.metrics import precision_recall_curve, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score

from skimage.filters import threshold_otsu

print "Completed imports"