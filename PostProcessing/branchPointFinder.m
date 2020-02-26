function [ branchpointMatrix ] = branchPointFinder( originalImage )
%This function uses a 9x9 window to find TRUE branch points
%The function defines a true branch point when the sum of the 9x9 window is
%greater than 13
%The function then searches through the branch points to find the ones that
%are too close together and decides that the true branch point is the
%middle point between close (less than 4 pixels) branch points


%Steps
    %Create Structuring Window (9x9) - possibly editable in future
    %Loop over original image
        %Sum for each window
        %If sum is greater than 13 - mark branch point
        %If not - do not mark
    %Loop over branchpoints
        %Use a 5x5 window
        %If sum of window is greater than 2
            %Find location of the branchpoints and find floor of average
            %value of x and y
    %Output branchpoint location matrix ( matrix of 0s and 1s the same size
    %as original image
    
    
    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    
branchpointMatrix = zeros(size(originalImage));

window = ones(9,9);
[row , col ] = size(originalImage);

originalImage = im2double(originalImage);

%loop through Image
for i = 5:row-5
    for j = 5:col-5
        imageROI = window.*originalImage(i-4:i+4,j-4:j+4);
        sumROI_image = sum(imageROI(:));
        if sumROI_image >= 13 && originalImage(i,j) == 1 
            branchpointMatrix(i,j) = 1;
        end
    end
end



window2 = ones(9,9);
%loop through branchpoints
for i = 5:row-5
    for j = 5:col-5
        branchptROI = window2.*branchpointMatrix(i-4:i+4,j-4:j+4);
        sumROI = sum(branchptROI(:));
        if sumROI >= 2
            [pointsX , pointsY] = find(branchptROI == 1);
            xloc = floor(mean(pointsX));
            yloc = floor(mean(pointsY));
            branchpointMatrix(i-4:i+4,j-4:j+4) = 0;
            branchpointMatrix(xloc, yloc) = 1;
        end
    end
end



end

