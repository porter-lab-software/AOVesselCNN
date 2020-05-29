% ComputeBatchVesselMetrics
% Gwen Musial & Hope Queener

%Segmentation Evaluation Code

% The purpose of this code is to compute the Dice coefficient between 
% The CNN segmentation result and the ground truth result, or
% any two manual segmentations
% Also calculates the density of the capillaries after canvas pixels are subtracted
% Accuracy, Dice and Modified Dice metrics assume that the 
% second image is the ground truth image

% The modified Dice allows tolerance for vessels that are identified
% along similar but not perfectly overlapping tracks

% Input Images are assumed to be single channel grayscale with 
% binary values of 0 and 255
% Folders are assumed to sort in corresponding order

% Sorensen-Dice coefficient   (DSC)
% 2(True Positive)/ [2TP + FP  +  FN]


close all

% Preference tags
first_folder_tag = 'FirstFolder';
second_folder_tag = 'SecondFolder';
canvas_folder_tag = 'CanvasFolder';

if ispref(mfilename,first_folder_tag)
    default_first_folder= getpref(mfilename,first_folder_tag);
else
    default_first_folder = '';
end
if ispref(mfilename,second_folder_tag)
    default_second_folder= getpref(mfilename,second_folder_tag);
else
    default_second_folder = '';
end
if ispref(mfilename,canvas_folder_tag)
    default_canvas_folder= getpref(mfilename,canvas_folder_tag);
else
    default_canvas_folder = '';
end
%% Select First set of images
[fileNames,pathName,fIndex] = uigetfile({'*.tif*'},...
    'Select file(s) to compare',fullfile(default_first_folder,'*.tif'), 'MultiSelect' , 'on');
if (fIndex == 0) % user pressed cancel
    fullPathName = '';
    return ; 
end
if ischar(fileNames)
    singleName = fileNames;
    fileNames = cell(1,1);
    fileNames{1} = singleName;
end
setpref(mfilename,first_folder_tag,pathName);

%% Read in Second Image
[fileNamesSecond,pathNameSecond,fIndex] = uigetfile({'*.tif*'},...
    'Select corresponding comparison file(s)',fullfile(default_second_folder,'*.tif'), 'MultiSelect', 'on');
if (fIndex == 0) % user pressed cancel
    return ; 
end
if ischar(fileNamesSecond)
    singleName = fileNamesSecond;
    fileNamesSecond = cell(1,1);
    fileNamesSecond{1} = singleName;
end
setpref(mfilename,second_folder_tag,pathNameSecond);
%% Read in Mask Image
[fileNamesCanvas,pathNameCanvas,fIndex] = uigetfile({'*.tif'},...
    'Select corresponding canvas images (canvas=255, non-canvas=0)',...
    fullfile(default_canvas_folder,'*.tif'),...
    'MultiSelect' , 'on');
if (fIndex == 0) % user pressed cancel
    return ; 
end
if ischar(fileNamesCanvas)
    singleName = fileNamesCanvas;
    fileNamesCanvas = cell(1,1);
    fileNamesCanvas{1} = singleName;
end
setpref(mfilename,canvas_folder_tag,pathNameCanvas);

fileCount = numel(fileNames);
if numel(fileNamesCanvas) ~= fileCount || ...
    numel(fileNamesSecond) ~= fileCount
    disp('Please select the same number of files.')
    return;
end

fileNames = sort(fileNames);
fileNamesSecond = sort(fileNamesSecond);
fileNamesCanvas = sort(fileNamesCanvas);

DensityImOne = nan(fileCount,1);
DensityImTwo = nan(fileCount,1);
UndilatedDice = nan(fileCount,1);
dilatedDice = nan(fileCount,1);
    
meanSegLength_Im1 = nan(fileCount,1);
meanSegLength_Im2 = nan(fileCount,1);
Acc = nan(fileCount,1);
Sensitivity = nan(fileCount,1);

for i = 1: fileCount
    originalImage = imread(fullfile(pathName,fileNames{i}));
    
    if ndims(originalImage) == 3
        fprintf('Please use single-channel input images with segmentation = 255 and background = 0.\n');
        originalImage = squeezes(originalImage(:,:,1));
    end
    binaryOriginal = (originalImage == max(originalImage(:)));
    
    comparisonImage = imread(fullfile(pathNameSecond,fileNamesSecond{i}));
    if ndims(comparisonImage) == 3
        fprintf('Please use single-channel input images with segmentation = 255 and background = 0.\n');
        comparisonImage = squeeze(comparisonImage(:,:,1));
    end
    binaryComparison = (comparisonImage == max(comparisonImage(:)));
    
    canvasImage = imread(fullfile(pathNameCanvas,fileNamesCanvas{i}));
    if ndims(canvasImage)==3
        fprintf('Please use single-channel canvas images with canvas = 255 and non-canvas = 0.\n');
        fprintf('%s\n',fullfile(pathName,fileNames{i}));
        canvasImage = squeeze(canvasImage(:,:,1));
    end
    if max(canvasImage(:))==0
        binaryCanvas = zeros(size(binaryOriginal)) == 1;
    else
        binaryCanvas = (canvasImage == max(canvasImage(:)));
    end
    
    %% Dilate both images
    
    se = strel('disk', 5);
    dilatedOriginal = imdilate(binaryOriginal, se);
    dilatedComparison = imdilate(binaryComparison,se);
    
    MandNN = and(dilatedOriginal,dilatedComparison);
    
    [rows,cols] = size(binaryOriginal);
    totalPixels = rows*cols;

    threeColorImage = zeros(rows,cols,3);
    threeColorImage(:,:,1) = binaryOriginal;
    threeColorImage(:,:,2) = binaryComparison;
    threeColorImage(:,:,3) = MandNN;
    
%     figure
%     imshow(threeColorImage)
%     
    
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
    
    dilatedDice(i) = (2*TP)/(2*TP + FN + FP);
    Acc_Dilated = (TP + TN_Dilated)/(TP + FN + FP + TN_Dilated);
    
    
    %% Undilated images
    undilatedIm(:,:,1) = binaryOriginal;
    undilatedIm(:,:,2) = binaryComparison;
    undilatedIm(:,:,3) = zeros(size(binaryOriginal));

    Red = undilatedIm(:,:,1) > 0 & undilatedIm(:,:,2) == 0 & undilatedIm(:,:,3) == 0;
    Red = sum(Red(:));
    Green = undilatedIm(:,:,1) == 0 & undilatedIm(:,:,2) > 0 & undilatedIm(:,:,3) == 0;
    Green = sum(Green(:));
    Yellow = undilatedIm(:,:,1) > 0 & undilatedIm(:,:,2) > 0 & undilatedIm(:,:,3) == 0;
    Yellow = sum(Yellow(:));
    
    TN = binaryOriginal(:,:) == 0 & binaryComparison(:,:) == 0;
    TN = sum(TN(:));
    
    % Dice No Dilation
    UndilatedDice(i) = (2*Yellow)/(2*Yellow + Red + Green);
    
    %% Accuracy
    % acc = (TP+TN)/(ALL)
    %Undilated
    Acc(i) = (Yellow + TN)/(totalPixels);
    
    
    %% Sensitivity
    %Sensitivity = TP / TP + FN
    Sensitivity(i) = Yellow/(Yellow + Red);
    
    % %% Get the Density of the Two Images
    capPixImOne = sum(binaryOriginal(:));
    capPixImTwo = sum(binaryComparison(:));
    canvasPixelCount = sum(binaryCanvas(:));
    DensityImOne(i) = capPixImOne/(totalPixels-canvasPixelCount);
    DensityImTwo(i) = capPixImTwo/(totalPixels-canvasPixelCount);
    
    %% Mean Segment Length
    
    %skeletonize images
    Skeleton_Im1 = bwmorph(originalImage ,'thin',Inf);
    Skeleton_Im2 = bwmorph(comparisonImage ,'thin',Inf);
    
    %remove spurs
    minLength = 15; 
    Skeleton_Im1 = removeSpurs(Skeleton_Im1,minLength);
    Skeleton_Im2 = removeSpurs(Skeleton_Im2,minLength);
    
    [meanSegLength_Im1_Val , stdLength_Im1] = MSL(Skeleton_Im1);
    [meanSegLength_Im2_Val , stdLength_Im2] = MSL(Skeleton_Im2);
    meanSegLength_Im1(i) = meanSegLength_Im1_Val;
    meanSegLength_Im2(i) = meanSegLength_Im2_Val;
end

%% Make a results Spreadsheet
headers = {'Image 1';
    'Image 2';
    'Canvas';
    'Density Image 1';
    'Density Image 2';
    'Dilated Dice';
    'MSL Image 1';
    'MSL Image 2';
    'Accuracy';
    'Sensitivity';
    'Dice'};
headers = headers'; % row-wise for xlswrite
columnCount = numel(headers);
[~,parentFolder] = fileparts(pathName(1:(end-1)));% get enclosing folder name
target_file = fullfile(pathName,strcat(parentFolder,'_Data.xls'));
dataMatrix = [DensityImOne DensityImTwo dilatedDice meanSegLength_Im1 meanSegLength_Im2 Acc Sensitivity UndilatedDice];
meanMetricRow = mean(dataMatrix,1);
stdMetricRow = std(dataMatrix,0,1);
rmsDensity = sqrt(sum((DensityImOne - DensityImTwo).^2)/fileCount);
rmsMSL = sqrt(sum((meanSegLength_Im1 - meanSegLength_Im2).^2)/fileCount);

% Headers
last_column = char('A'-1+columnCount);
xlswrite(target_file, headers,'sheet1',sprintf('A1:%c1',last_column));

%Write Data
last_row = fileCount + 1;
%xlswrite([pathName '\' fileName(1:length(fileName)-4) '_Data.xls'], tortuosity,'sheet1',['A2:A' num2str(length(tortuosity))]);
xlswrite(target_file, fileNames','sheet1',sprintf('A2:A%d',last_row));
xlswrite(target_file, fileNamesSecond','sheet1',sprintf('B2:B%d',last_row));
xlswrite(target_file, fileNamesCanvas','sheet1',sprintf('C2:C%d',last_row));
xlswrite(target_file, dataMatrix,'sheet1',sprintf('D2:%c%d', last_column,last_row));
mean_row = last_row+1;
xlswrite(target_file,{'Mean'},'sheet1',sprintf('A%d:A%d',mean_row,mean_row));
xlswrite(target_file,meanMetricRow,'sheet1',sprintf('D%d:%c%d',mean_row,last_column, mean_row));
std_row = mean_row+1;
xlswrite(target_file,{'Std'},'sheet1',sprintf('A%d:A%d',std_row,std_row));
xlswrite(target_file,stdMetricRow,'sheet1',sprintf('D%d:%c%d',std_row,last_column, std_row));
xlswrite(target_file,{'RMS'},'sheet1',sprintf('A%d:A%d',std_row+1,std_row+1));
xlswrite(target_file,rmsDensity,'sheet1',sprintf('D%d',std_row+1));
xlswrite(target_file,rmsMSL,'sheet1',sprintf('G%d',std_row+1));



