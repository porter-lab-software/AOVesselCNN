This folder contains the zipped data for the CNN training and validation.
 
Training

The zip files "Images01to17", "Images18to25" and "Imges28to31" are the grayscale images. 
These files should be unzipped and put into a single folder named 
[image_path]/Images, where image_path is defined in config_train.txt.

The TrainingGroundTruth.zip file provides the ground truth images, 
defining the 4 pixel classes, that are used during the training phase.
These files should be unzipped and put into a single folder named 
[image_path]/combined, where image_path is defined in config_train.txt.
 
The zip files ValidationGroundTruth and ValidationImages contain the 
validation grayscale and ground truth images used during validation.
These files should be unzipped and put into the folders 
[validation_image_path]/combined and [validation_image_path]/Images, 
where validation_image_path is defined in config_train.txt.

Segmentation (testing unknowns)

The zip file "example" contains the folder Images with 5 example grayscale 
perfusion images. The folder should be unzipped and put into the folder 
[image_path], such that the images are in [image_path]/Images,
where [image_path] is defined in config_test_only.txt. 

Metrics

The zip files CapillaryGroundTruth and ValidationCanvasImages are used with 
the MATLAB code in ./PostProcessing/VesselMetrics/ComputeBatchVesselMetrics.m  
