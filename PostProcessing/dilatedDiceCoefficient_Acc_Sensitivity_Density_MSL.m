%Gwen Musial

%Segmentation Evaluation Code

%Purpose of this code is to compute the Dice coefficient between manual
%segmented RPC images and a second image (either another manual
%segmentation or an automatic segmentation)

%Due to the nature of the vessel images, the images need to be dilated to
%give a better representation of the true segmentation overlap.

% Input Images MUST be BINARY

%Sorensen-Dice coefficient   (DSC)
% 2(True Positive)/ [2TP + FP  +  FN]

%Also calculates the density of the capillaries if a border image is input

close all

%% Read in First Image
userDirectoryPath = getenv('appdata');
settingsFilePath = fullfile(userDirectoryPath,mfilename);
if exist(settingsFilePath,'dir')
    recentFilePath = settingsFilePath;
else
    if mkdir(settingsFilePath) == 1
        recentFilePath = settingsFilePath;
    else
        errordlg('Could not create a settings folder for the current user.  Creating the settings file in the current directory.');
        recentFilePath = pwd;
    end
end
recentFileName = fullfile(recentFilePath,sprintf('%sSettings.mat',mfilename));
if exist(recentFileName,'file')
    load(recentFileName,'recentFile')
else
    recentFile = getenv('homepath');
end
[fileName,pathName,fIndex] = uigetfile({'*.*'},...
    'Select a file',recentFile);
if (fIndex == 0), % user pressed cancel
    fullPathName = '';
    return ; 
end

fullPathName = fullfile(pathName ,fileName);
recentFile = fullPathName; 
save(recentFileName,'recentFile');

originalImage = imread(fullfile(pathName,fileName));

if size(originalImage, 3) == 3
    originalImage = originalImage(:,:,1);
end

imageType = questdlg('Type of Capillary Image', ...
    'Capillary Input', ...
    'Black Vessels on White','White Vessels on Black','Cancel','Cancel');
switch imageType
    case 'Black Vessels on White'
        originalImage = imcomplement(originalImage);
end

%% Read in Second Image
[fileNameSecond,pathNameSecond,fIndex] = uigetfile({'*.*'},...
    'Select a Image for Comparison',recentFile);

comparisonImage = imread(fullfile(pathNameSecond,fileNameSecond));
if size(comparisonImage, 3) == 3
    comparisonImage = comparisonImage(:,:,1);
end

imageType = questdlg('Type of Capillary Image', ...
	'Capillary Input', ...
	'Black Vessels on White','White Vessels on Black','Cancel','Cancel');
switch imageType  
    case 'Black Vessels on White'
        comparisonImage = imcomplement(comparisonImage);
end

%% Dilate both images

se = strel('disk', 5);


dilatedOriginal = imdilate(originalImage, se);
dilatedComparison = imdilate(comparisonImage,se);

MandNN = and(dilatedOriginal,dilatedComparison);

[rows,cols] = size(originalImage);
threeColorImage = zeros(rows,cols,3);


threeColorImage(:,:,1) = originalImage;
threeColorImage(:,:,2) = comparisonImage;
threeColorImage(:,:,3) = MandNN;

%figure
%imshow(threeColorImage)


%% Sum pixels of each color
Red = threeColorImage(:,:,1) > 0 & threeColorImage(:,:,2) == 0 & threeColorImage(:,:,3) == 0;
Red = sum(Red(:));

Green = threeColorImage(:,:,1) == 0 & threeColorImage(:,:,2) > 0 & threeColorImage(:,:,3) == 0;
Green = sum(Green(:));

Blue = threeColorImage(:,:,1) == 0 & threeColorImage(:,:,2) == 0 & threeColorImage(:,:,3) >= 0;
Blue = sum(Blue(:));

White = threeColorImage(:,:,1) > 0 & threeColorImage(:,:,2) > 0 & threeColorImage(:,:,3) > 0;
TP_white = sum(White(:));

Yellow = threeColorImage(:,:,1) > 0 & threeColorImage(:,:,2) > 0 & threeColorImage(:,:,3) == 0;
Yellow = sum(Yellow(:));

Pink = threeColorImage(:,:,1) > 0 & threeColorImage(:,:,2) == 0 & threeColorImage(:,:,3) > 0;
Pink = sum(Pink(:));

Teal = threeColorImage(:,:,1) == 0 & threeColorImage(:,:,2) > 0 & threeColorImage(:,:,3) > 0;
Teal = sum(Teal(:));

Black = threeColorImage(:,:,1) == 0 & threeColorImage(:,:,2) == 0 & threeColorImage(:,:,3) == 0;
Black = sum(Black(:));

%% Dilated

TP = TP_white + Pink + Teal;
FN = Red;
FP = Green;
TN_Dilated = Blue+Black;

dilatedDice = (2*TP)/(2*TP + FN + FP)
Acc_Dilated = (TP + TN_Dilated)/(TP + FN + FP + TN_Dilated);


%% Undilated images
if max(originalImage(:)) == 1
    undilatedIm(:,:,1) = originalImage*255;
else
    undilatedIm(:,:,1) = originalImage;
end
if max(comparisonImage(:)) == 1
    undilatedIm(:,:,2) = comparisonImage*255;
else
    undilatedIm(:,:,2) = comparisonImage;
end

undilatedIm(:,:,3) = zeros(size(originalImage));
%figure
%imshow(undilatedIm)

Red = undilatedIm(:,:,1) > 0 & undilatedIm(:,:,2) == 0 & undilatedIm(:,:,3) == 0;
Red = sum(Red(:));
Green = undilatedIm(:,:,1) == 0 & undilatedIm(:,:,2) > 0 & undilatedIm(:,:,3) == 0;
Green = sum(Green(:));
Yellow = undilatedIm(:,:,1) > 0 & undilatedIm(:,:,2) > 0 & undilatedIm(:,:,3) == 0;
Yellow = sum(Yellow(:));

TN = originalImage(:,:) == 0 & comparisonImage(:,:) == 0;
TN = sum(TN(:));

% Dice No Dilation
UndilatedDice = (2*Yellow)/(2*Yellow + Red + Green)

%% Accuracy
% acc = (TP+TN)/(ALL)
%Undilated
totalPixels = rows*cols;
Acc = (Yellow + TN)/(totalPixels)


%% Sensitivity
%Sensitivity = TP / TP + FN
Sensitivity = Yellow/(Yellow + Red)

% %% Get the Density of the Two Images
% 
%Read in Mask Image
[fileNameMask,pathNameMask,fIndex] = uigetfile({'*.*'},...
    'Select a file',recentFile);
if (fIndex == 0), % user pressed cancel
    fullPathName = '';
    return ; 
end
maskImage = imread(fullfile(pathNameMask,fileNameMask));
maskType = questdlg('Type of Mask Image', ...
	'Mask Input', ...
	'Black Border','White Border','Cancel','Cancel');
switch maskType  
    case 'White Border'
        maskImage = imcomplement(maskImage);
end
[rows , cols] = size(comparisonImage);
totalPixels = rows*cols;
maskPixels = sum(maskImage(:) == min(maskImage(:)));
capPixImOne = sum(originalImage(:) == max(originalImage(:)));
capPixImTwo = sum(comparisonImage(:) == max(comparisonImage(:)));
DensityImOne = capPixImOne/(totalPixels-maskPixels)
DensityImTwo = capPixImTwo/(totalPixels-maskPixels)

%% Mean Segment Length

%skeletonize images
Skeleton_Im1 = bwmorph(originalImage ,'thin',Inf);
Skeleton_Im2 = bwmorph(comparisonImage ,'thin',Inf);

%remove spurs
minLength = 8;
Skeleton_Im1 = removeSpurs(Skeleton_Im1,minLength);
Skeleton_Im2 = removeSpurs(Skeleton_Im2,minLength);

[meanSegLength_Im1 , stdLength_Im1] = MSL(Skeleton_Im1)
[meanSegLength_Im2 , stdLength_Im2] = MSL(Skeleton_Im2)

%% Make a results Spreadsheet
Header1 = {'Density Image 1'};
Header2 = {'Density Image 2'};
Header3 = {'Dice'};
Header4 = {'Dilated Dice'};
Header5 = {'MSL Image 1'};
Header6 = {'MSL Image 2'};
Header7 = {'Accuracy'};
Header8 = {'Sensitivity'};

xlswrite([pathName '\' fileName(1:length(fileName)-4) '_Data.xls'], Header1,'sheet1','A1');
xlswrite([pathName '\' fileName(1:length(fileName)-4) '_Data.xls'], Header2,'sheet1','B1');
xlswrite([pathName '\' fileName(1:length(fileName)-4) '_Data.xls'], Header3,'sheet1','C1');
xlswrite([pathName '\' fileName(1:length(fileName)-4) '_Data.xls'], Header4,'sheet1','D1');

xlswrite([pathName '\' fileName(1:length(fileName)-4) '_Data.xls'], Header5,'sheet1','E1');
xlswrite([pathName '\' fileName(1:length(fileName)-4) '_Data.xls'], Header6,'sheet1','F1');
xlswrite([pathName '\' fileName(1:length(fileName)-4) '_Data.xls'], Header7,'sheet1','G1');
xlswrite([pathName '\' fileName(1:length(fileName)-4) '_Data.xls'], Header8,'sheet1','H1');

 %Write Data
xlswrite([pathName '\' fileName(1:length(fileName)-4) '_Data.xls'], DensityImOne,'sheet1','A2');
xlswrite([pathName '\' fileName(1:length(fileName)-4) '_Data.xls'], DensityImTwo,'sheet1','B2');
xlswrite([pathName '\' fileName(1:length(fileName)-4) '_Data.xls'], UndilatedDice,'sheet1','C2');
xlswrite([pathName '\' fileName(1:length(fileName)-4) '_Data.xls'], dilatedDice,'sheet1','D2');
 
xlswrite([pathName '\' fileName(1:length(fileName)-4) '_Data.xls'], meanSegLength_Im1,'sheet1','E2');
xlswrite([pathName '\' fileName(1:length(fileName)-4) '_Data.xls'], meanSegLength_Im2,'sheet1','F2');
xlswrite([pathName '\' fileName(1:length(fileName)-4) '_Data.xls'], Acc,'sheet1','G2');
xlswrite([pathName '\' fileName(1:length(fileName)-4) '_Data.xls'], Sensitivity,'sheet1','H2');
 