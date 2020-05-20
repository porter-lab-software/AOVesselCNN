function ThreshAndChopValidation
%Gwen Musial & Hope Queener
%Summer 2018, Spring 2020
%Code to reformat validation data generated by neural network
% Assumes height is the dimension of the square validation images
% Assumes that input image is uint8 with range 0 to 255
expectedPatchHeight= 768;
minimumObjectArea = 25;

%% Read in Input Image
close all;
recentFileTag = 'RecentFile';
if ispref(mfilename,recentFileTag)
    recentFilePath = getpref(mfilename,recentFileTag);
else
    recentFilePath = pwd;
end

[fileName,pathName,fIndex] = uigetfile({'*.png;*.tif'},...
    'Select a file',recentFilePath);
if (fIndex == 0) % user pressed cancel
    return ; 
end
fullPathName = fullfile(pathName ,fileName);
setpref(mfilename,recentFileTag,fullPathName);
grayPath = fullfile(pathName,'grayscale');
if ~isfolder(grayPath)
    mkdir(grayPath)
end
fprintf('%s\n',pathName);
reply = questdlg('Generate Binaries?', ...
    'ImageType', ...
    'Yes','No','Cancel','Cancel');

switch reply
    case 'Yes'
        saveBinaries = true;
        bwPath = fullfile(pathName,'BW');
        if ~isfolder(bwPath)
            mkdir(bwPath);
        end
        
    case 'Cancel'
        return;
    otherwise
        saveBinaries = false;
        bwPath = '';
end

originalImage = imread(fullfile(pathName, fileName));

[rowCount, columnCount] = size(originalImage);
patchHeight = rowCount;
rowpatchNum = 1;
if patchHeight ~= expectedPatchHeight
    fprintf('Warning! Patch height is %d (expected height is %d)\n',patchHeight, expectedPatchHeight);
end
patchWidth = patchHeight;
colpatchNum = floor(columnCount/patchWidth);
if mod(columnCount,patchWidth)~=0
    fprintf('The validation set is expected to be an integer number of square patches.');
    return;
end

% apply Otsu's method to binarize entire probability map
if saveBinaries
    if max(originalImage(:))< 200
        fprintf('Warning! Image may not have sufficient dynamic range of 0 to 255');
    end
    hist256 = imhist(originalImage,256);
    otsuLevel = otsuthresh(hist256);
    fprintf('Otsu threshold is %f\n',otsuLevel);
    binaryImage = imbinarize(originalImage,otsuLevel);
    labeledI = bwlabel(binaryImage);
    area = struct2cell(regionprops(logical(binaryImage),'area'));
    smallObjectsRemoved = binaryImage;
    for l = 1:length(area)
        if area{l} <= minimumObjectArea
            smallObjectsRemoved(labeledI == l) = 0;
        end
    end

else
    smallObjectsRemoved = [];
end

[~ , baseName ] = fileparts(fileName);
for k = 1:rowpatchNum
    rowStartNewImage = k*patchWidth-patchWidth+1;
    rowEndNewImage   = k*patchWidth;
    for j = 1:colpatchNum
        colStartNewImage = j*patchHeight - patchHeight +1;
        colEndNewImage   = j*patchHeight;
        newImagePatch = originalImage(rowStartNewImage:rowEndNewImage , colStartNewImage:colEndNewImage);
        imwrite(newImagePatch , fullfile(grayPath, sprintf( '%s_%02d_%02d.tif',baseName , k,j)));
        if saveBinaries
            binaryPatch = smallObjectsRemoved(rowStartNewImage:rowEndNewImage , colStartNewImage:colEndNewImage);
            imwrite(binaryPatch , fullfile(bwPath, sprintf( '%s_%02d_%02d_BW.tif',baseName , k,j)));
        end
    end
end