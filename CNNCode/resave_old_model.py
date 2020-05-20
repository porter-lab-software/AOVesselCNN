###################################################
#
#   Script to load and re-save model produced from prior tensorflow/keras package
#
##################################################

import os, sys
import ConfigParser
import numpy as np

# Keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Input, Concatenate, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Permute
from tensorflow.keras.layers import Activation, Flatten, Dense, ZeroPadding2D, AveragePooling2D, BatchNormalization , Reshape
from tensorflow.keras.models import Model , Sequential, model_from_json
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import categorical_accuracy, binary_accuracy
from tensorflow.keras.losses import categorical_crossentropy , binary_crossentropy

#Custon Functions
from RPC_CNN_Functions import *

from scipy import ndimage


sys.path.insert(0, './lib/')

print sys.argv
print "before reading configuraton file"

configFile = sys.argv[1]
configFilePath = os.path.join('.',configFile)

#config file to read from
config = ConfigParser.RawConfigParser()
config.readfp(open(configFilePath))
print configFilePath
#Load settings from Config file
config.read(configFilePath)

#path to the saved models (architecture.json, _best_weights.h5)
older_model_folder = config.get('data paths', 'older_model_folder')
model_file = config.get('data paths','model_file')
weights_file = config.get('data paths','weights_file')
newer_model_folder = config.get('data paths','newer_model_folder')

older_model_path = os.path.join(older_model_folder,model_file);
best_weights_path = os.path.join(older_model_folder,weights_file);
print older_model_path
print best_weights_path
model = model_from_json(open(older_model_path).read())
model.load_weights(best_weights_path)
model.compile(optimizer='rmsprop', loss=categorical_crossentropy,metrics=['categorical_accuracy'])
chars_per_line = 80;
print model.summary()

newer_model_path = os.path.join(newer_model_folder,'new_' + model_file);
print  newer_model_path
json_string = model.to_json()
open(newer_model_path, 'w').write(json_string)
print model.summary()