

function [ outputImage ] = removeSpurs( originalImage , minLength )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


branchPoints = branchPointFinder(originalImage);
branchPoints = imdilate(branchPoints,strel('disk',1));
branchPointsMatlab = bwmorph(originalImage,'branchpoints');


branchMask = originalImage &~ branchPoints;

%minLength = 15;
segmentImage = bwareaopen(branchMask, minLength,8);
%figure('Name' , 'Segment Image')
%imshow(segmentImage)

se = strel('disk',1);

% se1 = strel('line',3,0);
% se2 = strel('line',3,90);
% composition = imdilate(1,[se1 se2],'full');

% len = 5;
% for i = 5:row-6
%     for j = 5:col-6
%         angle = direction(i,j);
%         angleDeg = radtodeg(angle);       
%         SEangle = strel('lin',len,angleDeg);
%         [seRow , seCol] = size(SEangle.Neighborhood);
%         rowWidth = floor(seRow/2);
%         colWidth = floor(seCol/2);
%         if rowWidth == 0
%             rowWidth = 1;
%         end
%         if colWidth == 0
%             colWidth = 1;
%         end
%         
%         window = FrangiImage(i-rowWidth:i+rowWidth,j-colWidth:j+colWidth);
%         result = SEangle.Neighborhood.*window;
%         FrangiImage_Dilate(i,j) = max(result(:));
%  
%     end 
% end



dilateImage = imdilate(segmentImage,se);
% figure ('Name' , 'Dilate SE 5')
% imshow(dilateImage)

closeImage = bwmorph(dilateImage, 'close', 3);
% figure ('Name' , 'Close 1')
% imshow(closeImage)

outputImage = bwmorph(closeImage ,'skel',Inf);
% figure ('Name' , 'output')
% imshow(outputImage)



end

