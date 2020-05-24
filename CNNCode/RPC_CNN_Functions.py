#######################################################
#
# Support functions for training and applying training
#
#######################################################
#
# Authors: Gwen Musial, Hope Queener
# Date: 5/11/2020
#
import os, sys
import numpy as np
import h5py
from PIL import Image
import tensorflow
from tensorflow import keras
from sklearn.metrics import confusion_matrix , f1_score, jaccard_similarity_score , roc_curve
from sklearn.metrics import precision_recall_curve, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score

from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter

#Gets image datasets into correct format for conversion to HDF5 - One Channel
def get_datasets_unknown(imgs_dir,Nimgs,height,width):
    imgs = np.empty((Nimgs,height,width))
    print imgs_dir
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        print "in first loop"
        files.sort()
        for i in range(len(files)):
            #original
            print "original image: " + files[i]
            pil_img = Image.open(os.path.join(imgs_dir,files[i]))
            imgs[i] = np.asarray(pil_img )
            pil_img.close()
    print("imgs max: " +str(np.max(imgs)))
    print("imgs min: " +str(np.min(imgs)))
    imgs = np.reshape(imgs,(Nimgs,1,height,width))
    print(imgs.shape)
    return imgs

#Gets image datasets into correct format for conversion to HDF5 - Multiple Classes
# image names are [case]_[identifier]_[tile_row]_[tile_column]
# combo names are [case]_[identifier]_[tile_row]_[tile_column]_[suffix]
# the folder must contain only the expected images and no other files
def get_datasets(imgs_dir,Combos_dir ,Nimgs , height, width):
    imgs = np.empty((Nimgs,height,width))
    Combos = np.empty((Nimgs,height,width))
    
    for entry in os.listdir(Combos_dir):
        if os.path.isfile(os.path.join(Combos_dir, entry)):
            name_parts = entry.split('.')
            combo_parts = name_parts[0].split('_')
            combo_id = "_" + combo_parts[1] + "_"
            combo_part_count = len(combo_parts)
            if combo_part_count < 5:
                combo_suffix = ""
            else:
                combo_suffix = "_" + combo_parts[4]
            break
    
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        files.sort()
        for i in range(len(files)):
            #original
            print "original image: " + files[i]
            pil_img = Image.open(os.path.join(imgs_dir,files[i]))
            imgs[i] = np.asarray(pil_img )
            pil_img.close()
            name_parts = files[i].split(".")
            img_parts = name_parts[0].split("_")
            img_case = img_parts[0]
            img_row = img_parts[2]
            img_column = img_parts[3]
 
            #Combination Matrices
            combo_name = img_case + combo_id + img_row + "_" + img_column + combo_suffix + ".tif"
            print("Combo name: " + combo_name)
            pil_Combo = Image.open(os.path.join(Combos_dir, combo_name))
            Combos[i] = np.asarray(pil_Combo)
            pil_Combo.close()

    print("imgs max: " +str(np.max(imgs)))
    print("imgs min: " +str(np.min(imgs)))
    print("Combo max: " +str(np.max(Combos)))
    print("Combo min: " +str(np.min(Combos)))

    imgs = np.reshape(imgs,(Nimgs,1,height,width))
    Combos = np.reshape(Combos,(Nimgs,1,height,width))
    assert(imgs.shape == (Nimgs,1,height,width))
    print(imgs.shape)
    assert(Combos.shape == (Nimgs,1,height,width))
    return imgs, Combos

#Load the original data and return the extracted patches for training
def get_data_train_combo(train_imgs_original,
                      train_Combo,
                      patch_height,
                      patch_width,
                      stride):
    train_imgs = load_hdf5(train_imgs_original)
    train_Combo = load_hdf5(train_Combo)
    train_border = np.empty((train_Combo.shape))
    train_border[:] = 0
    train_border_logicals = (train_Combo == 2)
    train_border[train_border_logicals] = 1
    
    data_consistency_check(train_imgs,train_Combo)

    #check groundTruths are within 0-1
    print "max of Combo Images"
    print np.max(train_Combo)
    assert(np.min(train_Combo)==0 and np.max(train_Combo)==3)
    print "\ntrain images/masks shape:"
    print train_imgs.shape
    print "length images"
    print len(train_imgs)
    print "train images range (min-max): " +str(np.min(train_imgs)) +' - '+str(np.max(train_imgs))
    print "combination images range (min-max): " + str(np.min(train_Combo)) + '-'+str(np.max(train_Combo))

    
    input_pixel_count = float(train_Combo.size)
    print "\nCombo input pixel count = " + str(input_pixel_count)
    capillary_count = np.sum(train_Combo == 1)
    print "Percent capillaries:"
    print capillary_count/input_pixel_count
    large_vessel_count = np.sum(train_Combo == 3)
    print "Percent large vessels"
    print large_vessel_count / input_pixel_count
    border_pixel_count = np.sum(train_Combo == 2)
    print "Percent border/canvas pixels"
    print border_pixel_count / input_pixel_count
    background_pixel_count = np.sum(train_Combo == 0)
    print "Percent background pixels"
    print background_pixel_count / input_pixel_count
    
    #extract ORDERED OVERLAPPING TRAINING patches from the full images
    patches_imgs_train,patches_combo_train, iter_tot= extract_ordered_overlap_train_combo(train_imgs, train_Combo,patch_height,patch_width,stride,stride)

    data_consistency_check(patches_imgs_train,patches_combo_train)
    print "\n number of patches:" + str(iter_tot)

    return patches_imgs_train, patches_combo_train 

# Load the original data and return the extracted patches for testing
def get_data_testing_overlap(test_imgs_original, test_combo, imgs_to_test, 
    patch_height, patch_width, stride_height, stride_width):
    ### test
    test_imgs_original = load_hdf5(test_imgs_original)
    test_combo = load_hdf5(test_combo)

    test_imgs = test_imgs_original
    #extend both images and masks so they can be divided exactly by the patches dimensions
    test_imgs = test_imgs[0:imgs_to_test,:,:,:]
    test_imgs = paint_border_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)
    test_combo = test_combo[0:imgs_to_test,:,:,:]

    print "test images range (min-max): " +str(np.min(test_imgs)) +' - '+str(np.max(test_imgs))
   
    #extract the TEST patches from the full images
    patches_imgs_test = extract_ordered_overlap_test(test_imgs,patch_height,patch_width,stride_height,stride_width)
    #extract the TEST ground truth for all classes from the full images
    patches_combo_test = extract_ordered_overlap_test(test_combo,patch_height,patch_width,stride_height,stride_width)

    print "\ntest PATCHES images shape:"
    print patches_imgs_test.shape
    print "test PATCHES images range (min-max): " +str(np.min(patches_imgs_test)) +' - '+str(np.max(patches_imgs_test))

    return patches_imgs_test, test_imgs.shape[2], test_imgs.shape[3],  test_imgs, test_combo

# Load the original data and return the extracted patches for testing with no ground truth	
def get_data_unknown_Imgs(test_imgs, imgs_to_test, patch_height, patch_width, stride_height, stride_width):
    ### test
    # test_imgs = load_hdf5(test_imgs_original)
    print('in get_data_unknown_Imgs')
    print(test_imgs.shape)
    max_image_index = test_imgs.shape[0]
    if max_image_index > (imgs_to_test-1):
        max_image_index = imgs_to_test-1
    #extend both images and masks so they can be divided exactly by the patches dimensions
    if max_image_index == 0:
        test_imgs = test_imgs
    else:
        test_imgs = test_imgs[0:max_image_index,:,:,:]
    print('after select dim 0')
    print(test_imgs.shape)
    test_imgs = paint_border_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)
    print('after paint_border_overlap')
    print(test_imgs.shape)
    #extract the TEST patches from the full images
    patches_imgs_test = extract_ordered_overlap_test(test_imgs,patch_height,patch_width,stride_height,stride_width)
    print "\ntest PATCHES images shape:"
    print patches_imgs_test.shape
    print "test PATCHES images range (min-max): " +str(np.min(patches_imgs_test)) +' - '+str(np.max(patches_imgs_test))
    return patches_imgs_test, test_imgs.shape[2], test_imgs.shape[3]


#data consistency check
def data_consistency_check(imgs,masks):
    assert(len(imgs.shape)==len(masks.shape))
    assert(imgs.shape[0]==masks.shape[0])
    assert(imgs.shape[2]==masks.shape[2])
    assert(imgs.shape[3]==masks.shape[3])
    assert(masks.shape[1]==1)
    assert(imgs.shape[1]==1 or imgs.shape[1]==3)

#Adds borders
def paint_border_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert (len(full_imgs.shape)==4)  #4D arrays
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    leftover_h = (img_h-patch_h)%stride_h  #leftover on the h dim
    leftover_w = (img_w-patch_w)%stride_w  #leftover on the w dim
    if (leftover_h != 0):  #change dimension of img_h
        print "\nthe side H is not compatible with the selected stride of " +str(stride_h)
        print "img_h " +str(img_h) + ", patch_h " +str(patch_h) + ", stride_h " +str(stride_h)
        print "(img_h - patch_h) MOD stride_h: " +str(leftover_h)
        print "So the H dim will be padded with additional " +str(stride_h - leftover_h) + " pixels"
        tmp_full_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],img_h+(stride_h-leftover_h),img_w))
        tmp_full_imgs[0:full_imgs.shape[0],0:full_imgs.shape[1],0:img_h,0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    if (leftover_w != 0):   #change dimension of img_w
        print "the side W is not compatible with the selected stride of " +str(stride_w)
        print "img_w " +str(img_w) + ", patch_w " +str(patch_w) + ", stride_w " +str(stride_w)
        print "(img_w - patch_w) MOD stride_w: " +str(leftover_w)
        print "So the W dim will be padded with additional " +str(stride_w - leftover_w) + " pixels"
        tmp_full_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],full_imgs.shape[2],img_w+(stride_w - leftover_w)))
        tmp_full_imgs[0:full_imgs.shape[0],0:full_imgs.shape[1],0:full_imgs.shape[2],0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    print "new full images shape: \n" +str(full_imgs.shape)
    return full_imgs

#Divide all the full_imgs in patches
def extract_ordered_overlap_train_combo(full_imgs, full_Combo, patch_h, patch_w,stride_h,stride_w):
    assert (len(full_imgs.shape)==4)  #4D arrays
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    print img_h
    print patch_h
    print stride_h
    assert ((img_h-patch_h)%stride_h==0 and (img_w-patch_w)%stride_w==0)
    N_patches_img = ((img_h-patch_h)//stride_h+1)*((img_w-patch_w)//stride_w+1)  #// --> division between integers
    N_patches_tot = N_patches_img*full_imgs.shape[0]

    print "number of patches per image: " +str(N_patches_img) +", totally for this dataset: " +str(N_patches_tot)
    patches_Img = np.empty((N_patches_tot,full_imgs.shape[1],patch_h,patch_w))

    patches_Combo = np.empty((N_patches_tot,full_imgs.shape[1],patch_h,patch_w))
    iter_tot = 0   #iter over the total number of patches (N_patches)
    threshold = (patch_h*patch_h)/2
    print "threshold value"
    print threshold
    skipped_patch_count = 0
    for i in range(full_imgs.shape[0]):  #loop over the full images
        for h in range((img_h-patch_h)//stride_h+1):
            offset_h = h*stride_h
            for w in range((img_w-patch_w)//stride_w+1):
                offset_w = w*stride_w
                patch_Combo = full_Combo[i,:,offset_h:offset_h+patch_h,offset_w:(offset_w)+patch_w]
                nonborder_logicals = (patch_Combo != 2)
                nonborder_count = np.sum((nonborder_logicals)) 
                if nonborder_count <= threshold:
                    skipped_patch_count += 1
                else:
                    patch_Img = full_imgs[i,:,offset_h:(offset_h)+patch_h,offset_w:(offset_w)+patch_w]
                    # # patch_Img = gaussian_filter(patch_Img, sigma = 3)
                    patches_Img[iter_tot]=patch_Img
                    patches_Combo[iter_tot] = patch_Combo
                    iter_tot +=1   #total

    #Reshape the patches to only contain the total number of iterations that passed the border threshold
    print "total above threshold"
    print iter_tot
    print "total skipped"
    print skipped_patch_count
    
    reShapePatches_Im = np.empty((iter_tot,full_imgs.shape[1],patch_h,patch_w))
    reShapePatches_Combo =np.empty((iter_tot,full_imgs.shape[1],patch_h,patch_w))
    for i in range(iter_tot):
        reShapePatches_Im[i,:,:,:] = patches_Img[i,:,:,:]
        reShapePatches_Combo[i,:,:,:] = patches_Combo[i,:,:,:]
    print "different method reshape"
    print reShapePatches_Im.shape

    print "shape of patches array"
    print patches_Img.shape

    return reShapePatches_Im,  reShapePatches_Combo, iter_tot #array with all the full_imgs divided in patches

#Recompone the full images with the patches
def recompone_overlap(preds, img_h, img_w, stride_h, stride_w):
    assert (len(preds.shape)==4)  #4D arrays
    assert (preds.shape[1]==1 or preds.shape[1]==3)  #check the channel is 1 or 3
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    N_patches_h = (img_h-patch_h)//stride_h+1
    N_patches_w = (img_w-patch_w)//stride_w+1
    N_patches_img = N_patches_h * N_patches_w
    assert (preds.shape[0]%N_patches_img==0)
    N_full_imgs = preds.shape[0]//N_patches_img
    full_prob = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))  #itialize to zero mega array with sum of Probabilities
    full_sum = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))

    k = 0 #iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h-patch_h)//stride_h+1):
            offset_h = h*stride_h
            for w in range((img_w-patch_w)//stride_w+1):
                offset_w = w*stride_w
                full_prob[i,:,offset_h:(offset_h)+patch_h,offset_w:(offset_w)+patch_w]+=preds[k]
                full_sum[i,:,offset_h:(offset_h)+patch_h,offset_w:(offset_w)+patch_w]+=1
                k+=1
    assert(k==preds.shape[0])
    assert(np.min(full_sum)>=1.0)  #at least one
    final_avg = full_prob/full_sum
    assert(np.max(final_avg)<=1.0) #max value for a pixel is 1.0
    assert(np.min(final_avg)>=0.0) #min value for a pixel is 0.0
    return final_avg

#Divides full testing images to patches for NN to test on
def extract_ordered_overlap_test(full_imgs, patch_h, patch_w,stride_h,stride_w):
    assert (len(full_imgs.shape)==4)  #4D arrays
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    assert ((img_h-patch_h)%stride_h==0 and (img_w-patch_w)%stride_w==0)
    N_patches_img = ((img_h-patch_h)//stride_h+1)*((img_w-patch_w)//stride_w+1)  #// --> division between integers
    N_patches_tot = N_patches_img*full_imgs.shape[0]
    print "number of patches per image: " +str(N_patches_img) +", totally for this dataset: " +str(N_patches_tot)
    patches = np.empty((N_patches_tot,full_imgs.shape[1],patch_h,patch_w))
    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        for h in range((img_h-patch_h)//stride_h+1):
            offset_h = h*stride_h
            for w in range((img_w-patch_w)//stride_w+1):
                offset_w = w*stride_w
                patch = full_imgs[i,:,offset_h:(offset_h)+patch_h,offset_w:(offset_w)+patch_w]
                #patch = gaussian_filter(patch, sigma = 3)
                patches[iter_tot]=patch
                iter_tot +=1   #total
    assert (iter_tot==N_patches_tot)
    return patches  #array with all the full_imgs divided in patches

def load_hdf5(infile):
  with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
    return f["image"][()]

def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)


def group_images(data,per_row):
    #assert data.shape[0]%per_row==0
    assert (data.shape[1]==1 or data.shape[1]==3)
    data = np.transpose(data,(0,2,3,1))  #corect format for imshow
    all_stripe = []
    for i in range(int(data.shape[0]/per_row)):
        stripe = data[i*per_row]
        for k in range(i*per_row+1, i*per_row+per_row):
            stripe = np.concatenate((stripe,data[k]),axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1,len(all_stripe)):
        totimg = np.concatenate((totimg,all_stripe[i]),axis=0)
    return totimg

def visualize(data,filename):
    assert (len(data.shape)==3) #height*width*channels
    img = None
    if data.shape[2]==1:  #in case it is black and white
        data = np.reshape(data,(data.shape[0],data.shape[1]))
    if np.max(data)>1:
        img = Image.fromarray(data.astype(np.uint8))   #the image is already 0-255
        print(filename + ' is > 1')
    else:
        img = Image.fromarray((data*255).astype(np.uint8))  #the image is between 0-1
        print(filename + " is max of 1")
    img.save(filename + '.png')
    return img


def combinePredictions(pred_imgs_capillary,pred_imgs_LGVessel, pred_imgs_Border,pred_imgs_background):
    im_h = pred_imgs_capillary.shape[2]
    im_w = pred_imgs_capillary.shape[3]
    combinedPred = np.empty((pred_imgs_capillary.shape[0],im_h*im_w))
    pred_imgs_background = np.reshape(pred_imgs_background,(pred_imgs_capillary.shape[0],im_h*im_w))
    pred_imgs_Border = np.reshape(pred_imgs_Border,(pred_imgs_capillary.shape[0],im_h*im_w))
    pred_imgs_capillary = np.reshape(pred_imgs_capillary,(pred_imgs_capillary.shape[0],im_h*im_w))
    pred_imgs_LGVessel = np.reshape(pred_imgs_LGVessel,(pred_imgs_capillary.shape[0],im_h*im_w))
    for i in range(pred_imgs_capillary.shape[0]):
        for j in range(im_h*im_w):
            backgroundVal = (pred_imgs_background[i,j])
            borderVal = (pred_imgs_Border[i,j])
            capillaryVal = (pred_imgs_capillary[i,j])
            LGVesselVal = (pred_imgs_LGVessel[i,j])
            if capillaryVal > LGVesselVal and capillaryVal > borderVal: #Capillary
                combinedPred[i,j] = 1
            elif LGVesselVal > backgroundVal and LGVesselVal >capillaryVal and LGVesselVal > borderVal: #LGVessel
                combinedPred[i,j] = 3
            elif borderVal > backgroundVal and borderVal >capillaryVal and borderVal > LGVesselVal: #Border
                combinedPred[i,j] = 2
            elif backgroundVal > borderVal and backgroundVal >capillaryVal and backgroundVal > LGVesselVal: #Background
                combinedPred[i,j] = 0
            else:
                combinedPred[i,j] = 0
    return combinedPred


def pred_to_multiple_imgs(pred, patch_height, patch_width):
    print "number of classes in predictions"
    print pred.shape[2]  #check the classes are 4

    pred_images_0 = np.empty((pred.shape[0],pred.shape[1])) #(Npatches,height*width)
    pred_images_1 = np.empty((pred.shape[0],pred.shape[1])) #(Npatches,height*width)
    pred_images_2 = np.empty((pred.shape[0],pred.shape[1])) #(Npatches,height*width)
    pred_images_3 = np.empty((pred.shape[0],pred.shape[1])) #(Npatches,height*width)
    print "capillary predictions min and max"
    capillary_preds = pred[:,:,1]
    print np.amin(capillary_preds)
    print np.amax(capillary_preds)
    
    
    for i in range(pred.shape[0]):
        for pix in range(pred.shape[1]):
            pred_images_0[i,pix]=pred[i,pix,0]
            pred_images_1[i,pix]=pred[i,pix,1]
            pred_images_2[i,pix]=pred[i,pix,2]
            pred_images_3[i,pix]=pred[i,pix,3]
    pred_images_0 = np.reshape(pred_images_0,(pred_images_0.shape[0],1, patch_height, patch_width))
    pred_images_1 = np.reshape(pred_images_1,(pred_images_1.shape[0],1, patch_height, patch_width))
    pred_images_2 = np.reshape(pred_images_2,(pred_images_2.shape[0],1, patch_height, patch_width))
    pred_images_3 = np.reshape(pred_images_3,(pred_images_3.shape[0],1, patch_height, patch_width))
    return pred_images_0, pred_images_1, pred_images_2, pred_images_3



#Adds a hard-coded weight value to each class
def addWeightsToPatches(patches_combo_train):
    im_h = patches_combo_train.shape[2]
    im_w = patches_combo_train.shape[3]
    reShapedPatches = np.reshape(patches_combo_train,(patches_combo_train.shape[0],im_h*im_w))
    kerasLabeledPatchesArray = keras.utils.to_categorical(reShapedPatches, num_classes=4)
    reShapedWeights = reShapedPatches

    print "\nNew shape of traning patches:"
    print kerasLabeledPatchesArray.shape

    reShapedWeights[reShapedWeights == 0.0] = 1.0 # background
    reShapedWeights[reShapedWeights == 1.0] = 9.0 # capillary
    reShapedWeights[reShapedWeights == 2.0] = 1.0 # border
    reShapedWeights[reShapedWeights == 3.0] = 7.0 # large vessel
    return reShapedWeights , kerasLabeledPatchesArray

#Tells you the percent of each class in the training set
def combinationComposition(patches_combo_train,kerasLabeledPatchesArray):
    im_h = patches_combo_train.shape[2]
    im_w = patches_combo_train.shape[3]
    totalPixels = im_h*im_w*patches_combo_train.shape[0]

    print "\nTotalNumberofPixels"
    print totalPixels

    totalCapPixels = np.sum(kerasLabeledPatchesArray[:,:,1])
    percentCapPixels = totalCapPixels/totalPixels

    print "\nPercent CapillaryPixels"
    print percentCapPixels

    totalLVPixels = np.sum(kerasLabeledPatchesArray[:,:,3])
    percentLVPixels = totalLVPixels/totalPixels

    print "\nPercent Large Vessel Pixels"
    print percentLVPixels

    totalBorderPixels = np.sum(kerasLabeledPatchesArray[:,:,2])
    percentBorderPixels = totalBorderPixels/totalPixels

    print "\nPercent Border Pixels"
    print percentBorderPixels

    totalBGPixels = np.sum(kerasLabeledPatchesArray[:,:,0])
    percentBGPixels = totalBGPixels/totalPixels

    print "\nPercent Background Pixels"
    print percentBGPixels

#prints performance metrics
def displayPerformanceMetrics(y_true, y_thresh, y_prob, path_experiment):
    #Jaccard similarity index
    jaccard_index = jaccard_similarity_score(y_true, y_thresh, normalize=True)
    print "\nJaccard similarity score: " +str(jaccard_index)

    #F1 score
    F1_score = f1_score(y_true, y_thresh, labels=None, average='binary', sample_weight=None)
    print "\nF1 score (F-measure): " +str(F1_score)

    #Area under the ROC curve
    fpr, tpr, thresholds = roc_curve((y_true), y_prob)
    AUC_ROC = roc_auc_score(y_true, y_prob)
    print "\nArea under the ROC curve: " +str(AUC_ROC)

    #Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision = np.fliplr([precision])[0] 
    recall = np.fliplr([recall])[0] 
    AUC_prec_rec = np.trapz(precision,recall)
    print "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)

    #Save the results
    file_perf = open(path_experiment+'performances.txt', 'w')
    file_perf.write("Area under the ROC curve: "+str(AUC_ROC)
                + "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)
                + "\nJaccard similarity score: " +str(jaccard_index)
                + "\nF1 score (F-measure): " +str(F1_score)
)
    file_perf.close()
#From Confusion Matrix
def displayConfusionMatrixMetrics(confusion, path_experiment):
    print "just capillary confusion matrix"
    print confusion

    accuracy = 0
    if float(np.sum(confusion))!=0:
        accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
    print "Global Capillary Accuracy: " +str(accuracy)

    specificity = 0
    if float(confusion[0,0]+confusion[0,1])!=0:
        specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
    print "Specificity: " +str(specificity)

    sensitivity = 0
    if float(confusion[1,1]+confusion[1,0])!=0:
        sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
    print "Sensitivity: " +str(sensitivity)

    precision = 0
    if float(confusion[1,1]+confusion[0,1])!=0:
        precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
    print "Precision: " +str(precision)
    file_perf = open(path_experiment+'performances.txt', 'w')
    file_perf.write(
                "\n\nConfusion matrix:"
                +str(confusion)
                +"\nACCURACY: " +str(accuracy)
                +"\nSENSITIVITY: " +str(sensitivity)
                +"\nSPECIFICITY: " +str(specificity)
                +"\nPRECISION: " +str(precision)
                )
    file_perf.close()
