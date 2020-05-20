This folder contains the zipped data for the training and validation used when the code was developed.
 
The zip files "Images01to17", "Images18to25" and "Imges28to31" are the grayscale images used for training. 
These files should be unzipped and put into a single folder named "Images" within the image_path defined in config_train.txt.

The TrainingGroundTruth.zip file contains the folder "combined" which provides the ground truth images that are used during the training phase. 
The files in this folder, "combined" correspond to the training files in the "Images" and identify the 4 pixel classes in each image.

The zip files ValidationGroundTruth and ValidationImages contain the "combined" and "Images" folders required for validation after training.

All four path specifications are provided in the configuration file config_train.txt as input to the training.

The zip files CapillaryGroundTruth and ValidationCanvasImages are used with the MATLAB code in .\PostProcessing\VesselMetrics\ComputeBatchVesselMetrics.m  


