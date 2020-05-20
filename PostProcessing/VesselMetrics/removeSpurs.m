

function [ outputImage ] = removeSpurs( originalImage , minLength )
%First finds all of the branchpoints in a skeltonized image
%Then dilates the branchpoints
%Finds all of the segments in the image that are longer than the minimum
%length specified. 
%Performs morphological closing to "reconnect" segments

branchPoints = branchPointFinder(originalImage);
branchPoints = imdilate(branchPoints,strel('disk',1));

branchMask = originalImage &~ branchPoints;
segmentImage = bwareaopen(branchMask, minLength,8);

se = strel('disk',1);

dilateImage = imdilate(segmentImage,se);
closeImage = bwmorph(dilateImage, 'close', 3);
outputImage = bwmorph(closeImage ,'skel',Inf);

end

