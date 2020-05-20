function [meanSegLength , stdLength] = MSL(skeletonImage)
%function finds the branch points and endpoints in skeletonized Image
%Computes the length of the unbrached vessel segments
% March 2019

maxVal = max(skeletonImage(:));
[row , col ] = size(skeletonImage);
logicalOriginal = (skeletonImage==maxVal);

%Perform Vessel End detection on Image
vesselEnds = bwmorph(logicalOriginal, 'endpoints');
%Clean up not true end points
vesselEnds_before = vesselEnds;

for i = 4:row-4
    for j = 4:col-4
        if vesselEnds(i,j) == 1
            vesselEndWindow = logicalOriginal(i-1:i+1,j-1:j+1);
            sumVE = sum(vesselEndWindow(:));
            if sumVE == 2
                vesselEnds(i,j) = 1;
            elseif  4 <= sumVE   
                vesselEnds(i,j) = 0;
            else
                grownWindow = logicalOriginal(i-3:i+3, j-3:j+3);
                bpWindow = logicalOriginal(i-1:i+1,j-1:j+1);
                diffWindow = sum(grownWindow(:))- sum(bpWindow(:));
                if diffWindow >= 5
                    vesselEnds(i,j) = 0;
                end
            end
        end
    end
end

numEndsRemoved = sum(vesselEnds_before(:)) - sum(vesselEnds(:)); %#ok<NASGU>
totalEP = sum(vesselEnds(:)); %#ok<NASGU>
%perfrom branch point detection on skeleton image
branchPoints = bwmorph(logicalOriginal,'branch');

%Clean up not true branch points
branchPoints_before = branchPoints;

for i = 4:row-4
    for j = 4:col-4
        if branchPoints(i,j) == 1  
            bpWindow = logicalOriginal(i-1:i+1,j-1:j+1);
            sumBp = sum(bpWindow(:));
            sumMapSmall(i,j) = sumBp;
            
            if sumBp <= 3
                branchPoints(i,j) = 0;
            elseif  7 <= sumBp   
                branchPoints(i,j) = 1;
            else            
                grownWindow = logicalOriginal(i-3:i+3, j-3:j+3);
                diffWindow = sum(grownWindow(:))- sum(bpWindow(:));
                sumMapLarge(i,j) = sum(grownWindow(:));
                GW = skeletonImage(i-3:i+3, j-3:j+3);
                e = regionprops(grownWindow,'Eccentricity');
                eccVal = e.Eccentricity;
                eccentricityMap(i,j) = eccVal;
                p = regionprops(grownWindow,'Perimeter');
                perVal = p.Perimeter;
                perimeterMap(i,j) = p.Perimeter;
                if diffWindow <= 5 
                    branchPoints(i,j) = 0;
                end
                if eccVal >= 0.95 
                    branchPoints(i,j) = 0;
                end
            end
            
        end
    end
end

numBranchRemoved = sum(branchPoints_before(:)) - sum(branchPoints(:)); %#ok<NASGU>
totalBranchPoints = sum(branchPoints(:)); %#ok<NASGU>
bP = branchPoints;
branchPoints = imdilate(branchPoints,strel('disk',1));

branchMask = logicalOriginal &~ branchPoints;

Skeleton = logicalOriginal;

branchMask_EndMask = Skeleton &~ branchPoints &~ vesselEnds;

minLength = 5; %Based on Segment Length Analysis
%First connected components analysis
vesselSegs = bwareaopen(branchMask, minLength,8);
labeled = bwlabel(vesselSegs,8);
numSeg = max(labeled(:));
%% Compute MSL
pathlength = zeros(numSeg,1);

for i = 1:numSeg
    stats = regionprops(labeled(labeled == i) ,'Centroid','Perimeter', 'Orientation', 'Area');
    pathlength(i) = [stats(i).Area];       
end


%% Calulate average unbranched segment length 
% Need to add to pathlength to account for endpoint and branchpoint points

meanSegLength = mean(pathlength(:))+2;
stdLength = std(pathlength(:));


end

