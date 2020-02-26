%Gwen Musial
%Fall 2017
%Code to reformat data for use in training neural network

%Current data is in different sized images - need training data to be all
%the same dimensions

%Code will read in the image, the trace, the border, the large vessel image, and the combination image
%and output the reformatted data into their respective folders

%Code will divide the image into grids of 768x768 (a multiple of both 64
%and 128 so that the patch size used can be either one of these)


%% Read in Input Image

close all;
clear all;
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

[fileName,pathName,fIndex] = uigetfile({'*.tif'},...
    'Select a Red background original Image',recentFile);
if (fIndex == 0), % user pressed cancel
    fullPathName = '';
    return ; 
end

fullPathName = fullfile(pathName ,fileName);

% ...
% Do error checks on the file, if applicable
% Return without using file if error checks fail
% ... 

recentFile = fullPathName; %#ok<NASGU>
save(recentFileName,'recentFile');


originalImage = imread(fullfile(pathName, fileName));

%Determine border

if size(originalImage,3) == 3
    [ dilatedMask, invertedImage, numBackgroundPixels, redEdge] = removeBorder( originalImage );
    discSe = strel('disk',7);
    maskImage = (imdilate(dilatedMask,discSe));
    [numRows,numCols] = size(maskImage);
    for row = 1:numRows
        if sum(maskImage(row,:)~=0)
            topEdge = row;
            break
        end
    end
    for row = numRows:-1:1
        if sum(maskImage(row,:)~=0)
            BottomEdge = row;
            break
        end
    end
    for col = 1:numCols
        if sum(maskImage(:,col)~=0)
            rightEdge = col;
            break
        end
    end
    for col = numCols:-1:1
        if sum(maskImage(:,col)~=0)
            leftEdge = col;
            break
        end
    end

else
    [numRows,numCols] = size(originalImage);
    for row = 1:numRows
        if sum(originalImage(row,:)~=0)
            topEdge = row;
            break
        end
    end
    for row = numRows:-1:1
        if sum(originalImage(row,:)~=0)
            BottomEdge = row;
            break
        end
    end
    for col = 1:numCols
        if sum(originalImage(:,col)~=0)
            rightEdge = col;
            break
        end
    end
    for col = numCols:-1:1
        if sum(originalImage(:,col)~=0)
            leftEdge = col;
            break
        end
    end

end


%Crop OriginalImage
originalImageCropped = originalImage(topEdge:BottomEdge, rightEdge:leftEdge);

%Determine how many tiles to break image in to
[newNumRow,newNumCol] = size(originalImageCropped);

patchHeight = 768;
patchWidth  = 768;


%Determine how much padding to add to image to break into even patches
colpatchNum = floor(newNumCol/patchWidth);
if colpatchNum*patchWidth ~= newNumCol
    colpatchNum = colpatchNum+1;
    numColstoAdd = colpatchNum*patchWidth - newNumCol + colpatchNum -1;
else
    numColstoAdd = 0;
end

rowpatchNum = floor(newNumRow/patchHeight);
if rowpatchNum*patchHeight ~= newNumRow
    rowpatchNum = rowpatchNum+1;
    numRowstoAdd = (rowpatchNum)*patchHeight- newNumRow + rowpatchNum-1;
    
else
    numRowstoAdd = 0;
end

totalNewImages = colpatchNum*rowpatchNum;

%Add padding to images

paddedOriginalImage = padarray(originalImageCropped, [numRowstoAdd numColstoAdd] ,0, 'post');

%Break images into smaller patches and save outputs


i = 0;
for j = 1:colpatchNum
    for k = 1:rowpatchNum    
        rowStartNewImage = k*patchWidth-patchWidth+1;
        rowEndNewImage   = k*patchWidth;
        colStartNewImage = j*patchHeight - patchHeight +1;
        colEndNewImage   = j*patchHeight;
        newImagePatch = paddedOriginalImage(rowStartNewImage:rowEndNewImage , colStartNewImage:colEndNewImage );
        i = i+1;
        [~ , baseName ] = fileparts(fileName);
        imwrite(newImagePatch , fullfile(pathName, sprintf( '%s_%02d_%02d.tif',baseName , k,j)));
    end
end

%% Read in Mask Image
if exist('maskImage')
    %Crop Mask Image
    maskImageCropped = 255.*maskImage(topEdge:BottomEdge, rightEdge:leftEdge);
    %Add Padding to cropped Mask Image
    paddedMaskImage = padarray(maskImageCropped, [numRowstoAdd numColstoAdd] ,0, 'post');
    %Break images into smaller patches and save outputs
    i = 0;
    for j = 1:colpatchNum
        for k = 1:rowpatchNum
            rowStartNewImage = k*patchWidth-patchWidth+1;
            rowEndNewImage   = k*patchWidth;
            colStartNewImage = j*patchHeight - patchHeight +1;
            colEndNewImage   = j*patchHeight;
            newImagePatch    = paddedMaskImage(rowStartNewImage:rowEndNewImage , colStartNewImage:colEndNewImage );
            i = i+1;
            [~ , baseName ]  = fileparts(fileName);
            imwrite(newImagePatch , fullfile(pathName, sprintf( '%s_%02d_%02d_mask.gif',baseName , k,j)));
        end
    end
else
    

    [fileName_Mask,pathName_Mask,fIndex] = uigetfile({'*.gif'},...
        'Select a file',recentFile);
    if (fIndex == 0), % user pressed cancel
        fullPathName = '';
        return ;
    end
    
    fullPathName_Mask = fullfile(pathName_Mask,fileName_Mask);
    maskImage = imread(fullfile(pathName_Mask, fileName_Mask));
    
    %Crop Mask Image
    maskImageCropped = 255.*maskImage(topEdge:BottomEdge, rightEdge:leftEdge);
    %Add Padding to cropped Mask Image
    paddedMaskImage = padarray(maskImageCropped, [numRowstoAdd numColstoAdd] ,0, 'post');
    %Break images into smaller patches and save outputs
    i = 0;
    for j = 1:colpatchNum
        for k = 1:rowpatchNum
            rowStartNewImage = k*patchWidth-patchWidth+1;
            rowEndNewImage   = k*patchWidth;
            colStartNewImage = j*patchHeight - patchHeight +1;
            colEndNewImage   = j*patchHeight;
            newImagePatch    = paddedMaskImage(rowStartNewImage:rowEndNewImage , colStartNewImage:colEndNewImage );
            i = i+1;
            [~ , baseName ]  = fileparts(fileName_Mask);
            imwrite(newImagePatch , fullfile(pathName_Mask, sprintf( '%s_%02d_%02d.gif',baseName , k,j)));
        end
    end
    
end

%% Read in Skeleton Image


[fileName_Skel,pathName_Skel,fIndex] = uigetfile({'*.tif'},...
    'Select a Skeleton Capillary Image',recentFile);
if (fIndex == 0), % user pressed cancel
    fullPathName = '';
    return ; 
end

fullPathName_Skel = fullfile(pathName_Skel,fileName_Skel);
skeletonImage = imread(fullfile(pathName_Skel, fileName_Skel));

if length(size(skeletonImage)) == 3
    skeletonImage = skeletonImage(:,:,1);
end

if size(skeletonImage) ~= size(originalImage(:,:))
    disp('stop')
end



%Crop Skeleton Image
skeletonImageCropped = skeletonImage(topEdge:BottomEdge, rightEdge:leftEdge);

skeletonImageCropped(skeletonImageCropped == max(skeletonImageCropped(:))) = 255;
skeletonImageCropped(skeletonImageCropped == min(skeletonImageCropped(:))) = 0;

%skeletonImageCropped(skeletonImageCropped ~= 0) = 255;


%Add Padding to cropped Skeleton Image
paddedSkeletonImage = padarray(skeletonImageCropped, [numRowstoAdd numColstoAdd] ,0, 'post');



%Break images into smaller patches and save outputs
i = 0;
for j = 1:colpatchNum
    for k = 1:rowpatchNum    
        rowStartNewImage = k*patchWidth-patchWidth+1;
        rowEndNewImage   = k*patchWidth;
        colStartNewImage = j*patchHeight - patchHeight +1;
        colEndNewImage   = j*patchHeight;
        newImagePatch    = paddedSkeletonImage(rowStartNewImage:rowEndNewImage , colStartNewImage:colEndNewImage );
        i = i+1;
        [~ , baseName ]  = fileparts(fileName_Skel);
        imwrite(newImagePatch , fullfile(pathName_Skel, sprintf( '%s_%02d_%02d.gif',baseName , k,j)));
    end
end



%% Read in large vessel Image

[fileName_LGV,pathName_LGV,fIndex] = uigetfile({'*.tif'},...
    'Select Large Vessel Image',recentFile);
if (fIndex == 0), % user pressed cancel
    fullPathName = '';
    return ; 
end

fullPathName_LGV = fullfile(pathName_LGV,fileName_LGV);
LGVImage = imread(fullfile(pathName_LGV, fileName_LGV));

%Crop Mask Image
LGVImageCropped = 255.*LGVImage(topEdge:BottomEdge, rightEdge:leftEdge);
%Add Padding to cropped Mask Image
paddedLGVImage = padarray(LGVImageCropped, [numRowstoAdd numColstoAdd] ,0, 'post');
%Break images into smaller patches and save outputs
i = 0;
for j = 1:colpatchNum
    for k = 1:rowpatchNum    
        rowStartNewImage = k*patchWidth-patchWidth+1;
        rowEndNewImage   = k*patchWidth;
        colStartNewImage = j*patchHeight - patchHeight +1;
        colEndNewImage   = j*patchHeight;
        newImagePatch    = paddedLGVImage(rowStartNewImage:rowEndNewImage , colStartNewImage:colEndNewImage );
        i = i+1;
        [~ , baseName ]  = fileparts(fileName_LGV);
        imwrite(newImagePatch , fullfile(pathName_LGV, sprintf( '%s_%02d_%02d.gif',baseName , k,j)));
    end
end

%% Read in Combination Image


[fileName_Combo,pathName_Combo,fIndex] = uigetfile({'*.tif'},...
    'Select a Combination Image file',recentFile);
if (fIndex == 0), % user pressed cancel
    fullPathName = '';
    return ; 
end

fullPathName_Combo = fullfile(pathName_Combo,fileName_Combo);
ComboImage = imread(fullfile(pathName_Combo, fileName_Combo));

ComboImageCropped = ComboImage(topEdge:BottomEdge, rightEdge:leftEdge);
%Add Padding to cropped Mask Image
paddedComboImage = padarray(ComboImageCropped, [numRowstoAdd numColstoAdd] ,2, 'post');
%Break images into smaller patches and save outputs
i = 0;
for j = 1:colpatchNum
    for k = 1:rowpatchNum    
        rowStartNewImage = k*patchWidth-patchWidth+1;
        rowEndNewImage   = k*patchWidth;
        colStartNewImage = j*patchHeight - patchHeight +1;
        colEndNewImage   = j*patchHeight;
        newImagePatch    = paddedComboImage(rowStartNewImage:rowEndNewImage , colStartNewImage:colEndNewImage );
        i = i+1;
        [~ , baseName ]  = fileparts(fileName_Combo);
        imwrite(newImagePatch , fullfile(pathName_Combo, sprintf( '%s_%02d_%02d_Combined.tif',baseName , k,j )));
    end
end

