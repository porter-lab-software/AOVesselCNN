function ColocalizeImages

chCount = 3;
[recentFolder,recentFiles] = InitPreferences(mfilename);

if isempty(recentFiles)
    defaultFile = recentFolder;
else
    defaultFile = fullfile(recentFolder,recentFiles{1});
end

selectionOptions = {'*.tif;*.png; *.gif','Image files (*.tif, *.png, *.gif)';
   '*.*',  'All Files (*.*)'};
filePaths = cell(chCount,1);
for channelIndex = 1:chCount
    [fileName,pathName,fIndex] = uigetfile(selectionOptions,...
        'Select images for colocalization',defaultFile);
    if (fIndex == 0) 
        if channelIndex == 1
            return ; 
        else
            filePaths{channelIndex} = '';
        end
    else
        if channelIndex == 1
            recentFolder = pathName;
            recentFile = fileName;
        end
        defaultFile = recentFolder; % expect file names to change
        filePaths{channelIndex} = fullfile(pathName,fileName);
    end
end    
disp(pathName)

rgbImage = [];
imageHeight = zeros(chCount,1);
imageWidth = zeros(chCount,1);
titleMessage = '';
colorOrder = {'Red';'Green';'Blue'};
for channelIndex = 1:chCount
    if isempty(filePaths{channelIndex})
       imageHeight(channelIndex) = imageHeight(1);
       imageWidth(channelIndex) = imageWidth(1);
       imageMatrix = zeros(imageHeight(1),imageWidth(1));
    else    
        thisImage = imread(filePaths{channelIndex});
        switch(ndims(thisImage))
            case 2
                imageHeight(channelIndex) = size(thisImage,1);
                imageWidth(channelIndex) = size(thisImage,2);
                if isa(thisImage,'uint8') 
                    if (max(thisImage(:))>1)
                        imageMatrix = double(thisImage)/255;
                    else
                        imageMatrix = double(thisImage);
                    end
                elseif islogical(thisImage)
                    imageMatrix = double(thisImage);
                else
                    imageMatrix = zeros(imageHeight(channelIndex),imageWidth(channelIndex));
                end
            case 3
                if size(thisImage,3) == 3
                    if thisImage(:,:,1) == thisImage(:,:,2) 
                        imageHeight(channelIndex) = size(thisImage,1);
                        imageWidth(channelIndex) = size(thisImage,2);
                        if max(thisImage) > 1.0
                            imageMatrix = thisImage(:,:,1);
                        end
                    else
                        fprintf('Unrecognized RGB format. Unable to proceed.\n')
                        return;
                    end
                else
                    fprintf('Unrecognized RGB format. Unable to proceed.\n')
                    return;
                end

            otherwise
                fprintf('Number of dimensions of this image is %d. Unrecognized format.\n',ndims(imageMatrix));
                return;
        end
    end
    if channelIndex == 1
        rgbImage = zeros(imageHeight(1),imageWidth(1),3);
        rgbImage(:,:,1) = imageMatrix;
    else
        if (imageHeight(channelIndex) == imageHeight(1)) && ...
            (imageWidth(channelIndex) == imageWidth(1))
                rgbImage(:,:,channelIndex) = imageMatrix; %#ok<AGROW>
        end
    end
       
end

UpdatePreferences(mfilename,recentFolder,recentFile)

figure('Name','Co-localized Images','NumberTitle','off')
imagesc(rgbImage)
title(titleMessage,'interpreter','none')
for channelIndex = 1:chCount
    fprintf('%s: %s\n',colorOrder{channelIndex},filePaths{channelIndex})
end

rgDifference = rgbImage(:,:,1) - rgbImage(:,:,2);
rbDifference = rgbImage(:,:,1) - rgbImage(:,:,3);
fprintf('R-G Difference sum = %e\n',sum(rgDifference(:)));
fprintf('R-B Difference sum = %e\n',sum(rbDifference(:)));


function [recentFolder,recentFiles] = InitPreferences(applicationName)
recentFilesTag = 'RecentFiles';
recentFolderTag = 'RecentFolder';
if ispref(applicationName,recentFilesTag)
    recentFiles = getpref(applicationName,recentFilesTag);
else
    recentFiles = {};
end
if ispref(applicationName,recentFolderTag)
    recentFolder = getpref(applicationName,recentFolderTag);
    if ~exist(recentFolder,'file')
        recentFolder = pwd;
    end
else
    recentFolder = pwd;
end


function UpdatePreferences(applicationName,recentFolder,recentFile)
recentFilesTag = 'RecentFile';
recentFolderTag = 'RecentFolder';

setpref(applicationName,recentFilesTag,recentFile);
setpref(applicationName,recentFolderTag,recentFolder);
