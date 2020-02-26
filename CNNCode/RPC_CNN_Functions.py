import os, sys
import numpy as np
import h5py
from PIL import Image
import keras
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
        for i in range(len(files)):
            #original
            print "original image: " + files[i]
            img = Image.open(imgs_dir+files[i])
            imgs[i] = np.asarray(img)
    print("imgs max: " +str(np.max(imgs)))
    print("imgs min: " +str(np.min(imgs)))
    imgs = np.reshape(imgs,(Nimgs,1,height,width))
    print(imgs.shape)
    return imgs

#Gets image datasets into correct format for conversion to HDF5 - Multiple Classes
def get_datasets(imgs_dir,groundTruth_dir,borderMasks_dir,LGvesselMasks_dir,Combos_dir ,Nimgs , height, width, train_test="null"):
    imgs = np.empty((Nimgs,height,width))
    groundTruth = np.empty((Nimgs,height,width))
    border_masks = np.empty((Nimgs,height,width))
    LGvessel_masks = np.empty((Nimgs,height,width))
    Combos = np.empty((Nimgs,height,width))
    print imgs_dir
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        print "in first loop"
        for i in range(len(files)):
            #original
            print "original image: " + files[i]
            img = Image.open(imgs_dir+files[i])
            imgs[i] = np.asarray(img)
            #corresponding ground truth
            groundTruth_name = ""
            if train_test =="train":
                groundTruth_name = files[i][0:2] + "_manual1_" + files[i][12:17] + ".gif"
                print("ground truth name: " + groundTruth_name)
                g_truth = Image.open(groundTruth_dir + groundTruth_name)
                groundTruth[i] = np.asarray(g_truth)
            elif train_test=="test":
                groundTruth_name = files[i][0:2] + "_manual1_" + files[i][8:13] + ".gif"
                print("ground truth name: " + groundTruth_name)
                g_truth = Image.open(groundTruth_dir + groundTruth_name)
                groundTruth[i] = np.asarray(g_truth)
            elif train_test == "unknown":
                print("no ground truth")
            else:
                print("specify train or test")
                exit()
            #corresponding border masks
            border_masks_name = ""
            if train_test=="train":
                border_masks_name = files[i][0:2] + "_training_mask_" + files[i][12:17] + ".gif"
            elif train_test=="test":
                border_masks_name = files[i][0:2] + "_test_mask_" + files[i][8:13] + ".gif"
            elif train_test == "unknown":
                border_masks_name = files[i][0:19] + "_mask.gif" 
            else:
                print("specify if train or test!!")
                exit()
            print("border masks name: " + border_masks_name)
            b_mask = Image.open(borderMasks_dir + border_masks_name)
            border_masks[i] = np.asarray(b_mask)
            #corresponding Vessel masks
            vessel_masks_name = ""
            if train_test=="train":
                vessel_masks_name = files[i][0:2] + "_trainingLargeV_" + files[i][12:17] + ".tif"
                print("Vessel masks name: " + vessel_masks_name)
                LG_mask = Image.open(LGvesselMasks_dir + vessel_masks_name)
                tmp = np.asarray(LG_mask)
                if tmp.shape == (768,768,3):
                     tmp = tmp[:,:,1]
                LGvessel_masks[i] = tmp
            elif train_test=="test":
                vessel_masks_name = files[i][0:2] + "_testLargeV_" + files[i][8:13] + ".tif"
                print("Vessel masks name: " + vessel_masks_name)
                LG_mask = Image.open(LGvesselMasks_dir + vessel_masks_name)
                tmp = np.asarray(LG_mask)
                if tmp.shape == (768,768,3):
                     tmp = tmp[:,:,1]
                LGvessel_masks[i] = tmp
            elif train_test == "unknown":
                print("no vessel masks exist")
            else:
                print("specify if train or test!!")
                exit()

            #Combination Matrices
            combo_name = ""
            if train_test=="train":
                combo_name = files[i][0:2] + "_manual1_" + files[i][12:17] + "_Combined.tif"
                print("Combo name: " + combo_name)
                Combo = Image.open(Combos_dir + combo_name)
                Combos[i] = np.asarray(Combo)
            elif train_test=="test":
                combo_name = files[i][0:2] + "_manual1_" + files[i][8:13] + "_Combined.tif"
                print("Combo name: " + combo_name)
                Combo = Image.open(Combos_dir + combo_name)
                Combos[i] = np.asarray(Combo)
            elif train_test == "unknown":
                print("no combo matrix exists")
            else:
                print("specify if train or test!!")
                exit()

    print("imgs max: " +str(np.max(imgs)))
    print("imgs min: " +str(np.min(imgs)))
    print("GT max: " +str(np.max(groundTruth)))
    print("GT min: " +str(np.min(groundTruth)))
    print("Border max: " +str(np.max(border_masks)))
    print("Border min: " +str(np.min(border_masks)))
    print("LG max: " +str(np.max(LGvessel_masks)))
    print("LG min: " +str(np.min(LGvessel_masks)))
    print("Combo max: " +str(np.max(Combos)))
    print("Combo min: " +str(np.min(Combos)))

    imgs = np.reshape(imgs,(Nimgs,1,height,width))
    groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))
    border_masks = np.reshape(border_masks,(Nimgs,1,height,width))
    LGvessel_masks = np.reshape(LGvessel_masks,(Nimgs,1,height,width))
    Combos = np.reshape(Combos,(Nimgs,1,height,width))
    assert(imgs.shape == (Nimgs,1,height,width))
    print(imgs.shape)
    assert(groundTruth.shape == (Nimgs,1,height,width))
    assert(border_masks.shape == (Nimgs,1,height,width))
    assert(LGvessel_masks.shape == (Nimgs,1,height,width))
    assert(Combos.shape == (Nimgs,1,height,width))
    return imgs, groundTruth, border_masks , LGvessel_masks, Combos

#Load the original data and return the extracted patches for training
def get_data_train_combo(train_imgs_original,
                      train_border,
                      train_Combo,
                      patch_height,
                      patch_width,
                      stride):
    train_imgs = load_hdf5(train_imgs_original)
    train_border = load_hdf5(train_border)
    train_Combo = load_hdf5(train_Combo)

    if np.max(train_border) !=1:
        train_border = train_border/255.


    data_consistency_check(train_imgs,train_border)
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

    #extract ORDERED OVERLAPPING TRAINING patches from the full images
    patches_imgs_train,patches_border_train  ,patches_combo_train, iter_tot= extract_ordered_overlap_train_combo(train_imgs,train_border,train_Combo,patch_height,patch_width,stride,stride)

    data_consistency_check(patches_imgs_train,patches_combo_train)
    print "\n number of patches:" + str(iter_tot)

    return patches_imgs_train,patches_border_train , patches_combo_train 

# Load the original data and return the extracted patches for testing
def get_data_testing_overlap(test_imgs_original, test_Capillaries, test_border , 
    test_combo, test_LGVessels, imgs_to_test, patch_height, patch_width, stride_height, stride_width):
    ### test
    test_imgs_original = load_hdf5(test_imgs_original)
    test_Capillaries = load_hdf5(test_Capillaries)
    test_border = load_hdf5(test_border)
    test_combo = load_hdf5(test_combo)
    test_LGVessels = load_hdf5(test_LGVessels)

    #test_imgs = my_PreProc(test_imgs_original)
    test_imgs = test_imgs_original
    test_border = test_border/255.
    test_Capillaries = test_Capillaries/255.
    test_LGVessels = test_LGVessels/255.
    #extend both images and masks so they can be divided exactly by the patches dimensions
    test_imgs = test_imgs[0:imgs_to_test,:,:,:]
    test_border = test_border[0:imgs_to_test,:,:,:]
    test_Capillaries = test_Capillaries[0:imgs_to_test,:,:,:]
    test_imgs = paint_border_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)
    test_combo = test_combo[0:imgs_to_test,:,:,:]
    test_LGVessels = test_LGVessels[0:imgs_to_test,:,:,:]

    print "test images range (min-max): " +str(np.min(test_imgs)) +' - '+str(np.max(test_imgs))
   
    #extract the TEST patches from the full images
    patches_imgs_test = extract_ordered_overlap_test(test_imgs,patch_height,patch_width,stride_height,stride_width)
    #extract the TEST ground truth for all classes from the full images
    patches_capillary_test = extract_ordered_overlap_test(test_Capillaries,patch_height,patch_width,stride_height,stride_width)
    patches_border_test = extract_ordered_overlap_test(test_border,patch_height,patch_width,stride_height,stride_width)
    patches_LGV_test = extract_ordered_overlap_test(test_LGVessels,patch_height,patch_width,stride_height,stride_width)
    patches_combo_test = extract_ordered_overlap_test(test_combo,patch_height,patch_width,stride_height,stride_width)

    print "\ntest PATCHES images shape:"
    print patches_imgs_test.shape
    print "test PATCHES images range (min-max): " +str(np.min(patches_imgs_test)) +' - '+str(np.max(patches_imgs_test))

    return patches_imgs_test, test_imgs.shape[2], test_imgs.shape[3], test_Capillaries , test_border , test_combo,test_LGVessels , patches_capillary_test , patches_border_test , patches_LGV_test, patches_combo_test 

# Load the original data and return the extracted patches for testing with no ground truth	
def get_data_unknown_Imgs(test_imgs_original, imgs_to_test, patch_height, patch_width, stride_height, stride_width):
    ### test
    test_imgs = load_hdf5(test_imgs_original)
    #extend both images and masks so they can be divided exactly by the patches dimensions
    test_imgs = test_imgs[0:imgs_to_test,:,:,:]
    test_imgs = paint_border_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)
    #extract the TEST patches from the full images
    patches_imgs_test = extract_ordered_overlap_test(test_imgs,patch_height,patch_width,stride_height,stride_width)
    print "\ntest PATCHES images shape:"
    print patches_imgs_test.shape
    print "test PATCHES images range (min-max): " +str(np.min(patches_imgs_test)) +' - '+str(np.max(patches_imgs_test))
    return patches_imgs_test, test_imgs.shape[2], test_imgs.shape[3]

#data consinstency check
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
def extract_ordered_overlap_train_combo(full_imgs, full_borders , full_Combo, patch_h, patch_w,stride_h,stride_w):
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

    patches_Border = np.empty((N_patches_tot,full_imgs.shape[1],patch_h,patch_w))
    patches_Combo = np.empty((N_patches_tot,full_imgs.shape[1],patch_h,patch_w))
    iter_tot = 0   #iter over the total number of patches (N_patches)
    a = np.array([])
    threshold = (patch_h*patch_h)/4
    print "threshold value"
    print threshold
    for i in range(full_imgs.shape[0]):  #loop over the full images
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                patchBorder = full_borders[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]
                sumROI = np.sum(patchBorder)
                if sumROI <= threshold:
                    continue
                patch_Img = full_imgs[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]
                #patch_Img = gaussian_filter(patch_Img, sigma = 3)
                patch_Border = full_borders[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]
                patch_Combo = full_Combo[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]
                patches_Img[iter_tot]=patch_Img
                patches_Border[iter_tot] = patch_Border
                patches_Combo[iter_tot] = patch_Combo

                iter_tot +=1   #total

    #Reshape the patches to only contain the total number of iterations that passed the border threshold
    reShapePatches_Im = np.empty((iter_tot,full_imgs.shape[1],patch_h,patch_w))
    reShapePatches_Border = np.empty((iter_tot,full_imgs.shape[1],patch_h,patch_w))
    reShapePatches_Combo =np.empty((iter_tot,full_imgs.shape[1],patch_h,patch_w))
    for i in range(iter_tot):
        reShapePatches_Im[i,:,:,:] = patches_Img[i,:,:,:]
        reShapePatches_Border[i,:,:,:] = patches_Border[i,:,:,:]
        reShapePatches_Combo[i,:,:,:] = patches_Combo[i,:,:,:]
    print "different method reshape"
    print reShapePatches_Im.shape

    print "shape of patches array"
    print patches_Img.shape

    return reShapePatches_Im,  reShapePatches_Border, reShapePatches_Combo, iter_tot #array with all the full_imgs divided in patches

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
            for w in range((img_w-patch_w)//stride_w+1):
                full_prob[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=preds[k]
                full_sum[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=1
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
            for w in range((img_w-patch_w)//stride_w+1):
                patch = full_imgs[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]
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

#convert RGB image in black and white
def rgb2gray(rgb):
    assert (len(rgb.shape)==4)  #4D arrays
    assert (rgb.shape[1]==3)
    bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
    return bn_imgs

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
    else:
        img = Image.fromarray((data*255).astype(np.uint8))  #the image is between 0-1
    img.save(filename + '.png')
    return img

def saveIms( data, filename):
    totalIms = data.shape[0]
    print "total Images"
    print totalIms
    for imNum in range(data.shape[0]):
        singleImg = None
        singleIm = data[imNum,:,:,:]
        print "shape of single image"
        print singleIm.shape
        if singleIm.shape[2]==1:  #in case it is black and white
            singleIm = np.reshape(singleIm,(singleIm.shape[0],singleIm.shape[1]))
        if np.max(singleIm)>1:
            singleIm = Image.fromarray(singleIm.astype(np.uint8))   #the image is already 0-255
        else:
            singleIm = Image.fromarray((singleIm*255).astype(np.uint8))  #the image is between 0-1
        singleImg.save(filename + imNum + '.tif')
    return singleImg

def masks_to_reshape(masks):
    assert (len(masks.shape)==4)  #4D arrays
    assert (masks.shape[1]==1 )  #check the channel is 1
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    masks = np.reshape(masks,(masks.shape[0],im_h*im_w))
    new_masks = np.empty((masks.shape[0],im_h*im_w,2))
    for i in range(masks.shape[0]):
        for j in range(im_h*im_w):
            if  masks[i,j] == 0:
                new_masks[i,j,0]=1
                new_masks[i,j,1]=0
            else:
                new_masks[i,j,0]=0
                new_masks[i,j,1]=1
    return new_masks

def masks_to_reshape_4Class(masks):
    assert (len(masks.shape)==4)  #4D arrays
    assert (masks.shape[1]==1 )  #check the channel is 1
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    masks = np.reshape(masks,(masks.shape[0],im_h*im_w))
    new_masks = np.empty((masks.shape[0],im_h*im_w,4))
    for i in range(masks.shape[0]):
        for j in range(im_h*im_w):
            if (np.int(masks[i,j])) == np.int(0): #Background
                new_masks[i,j,0]=1
                new_masks[i,j,1]=0
                new_masks[i,j,2]=0
                new_masks[i,j,3]=0
            elif (np.int((masks[i,j]))) == np.int(2): #Border
                new_masks[i,j,0]=0
                new_masks[i,j,1]=0
                new_masks[i,j,2]=0
                new_masks[i,j,3]=0
            elif (np.int(masks[i,j])) == np.int(3): #Large Vessel
                new_masks[i,j,0]=0
                new_masks[i,j,1]=0
                new_masks[i,j,2]=0
                new_masks[i,j,3]=1
            elif np.int(masks[i,j]) == np.int(1): #Capillary
                new_masks[i,j,0]=0
                new_masks[i,j,1]=1
                new_masks[i,j,2]=0
                new_masks[i,j,3]=0
            else:
                print "inside else loop"
                print str(np.int(masks[i,j]))

    return new_masks

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

def pred_to_imgs(pred, patch_height, patch_width, mode="original"):
    assert (len(pred.shape)==3)  #3D array: (Npatches,height*width,2)
    assert (pred.shape[2]==2 )  #check the classes are 2
    pred_images = np.empty((pred.shape[0],pred.shape[1]))  #(Npatches,height*width)
    if mode=="original":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                pred_images[i,pix]=pred[i,pix,1]
    elif mode=="threshold":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i,pix,1]>=0.5:
                    pred_images[i,pix]=1
                else:
                    pred_images[i,pix]=0
    else:
        print "mode " +str(mode) +" not recognized, it can be 'original' or 'threshold'"
        exit()
    pred_images = np.reshape(pred_images,(pred_images.shape[0],1, patch_height, patch_width))
    return pred_images

def pred_to_multiple_imgs(pred, patch_height, patch_width, mode="original"):
    print "number of classes in predictions"
    print pred.shape[2]  #check the classes are 4

    pred_images_0 = np.empty((pred.shape[0],pred.shape[1])) #(Npatches,height*width)
    pred_images_1 = np.empty((pred.shape[0],pred.shape[1])) #(Npatches,height*width)
    pred_images_2 = np.empty((pred.shape[0],pred.shape[1])) #(Npatches,height*width)
    pred_images_3 = np.empty((pred.shape[0],pred.shape[1])) #(Npatches,height*width)

    if mode=="original":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                pred_images_0[i,pix]=pred[i,pix,0]
                pred_images_1[i,pix]=pred[i,pix,1]
                pred_images_2[i,pix]=pred[i,pix,2]
                pred_images_3[i,pix]=pred[i,pix,3]
    elif mode=="threshold":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i,pix,0]>=0.8:
                    pred_images_0[i,pix]=1
                if pred[i,pix,0]<=0.8:
                    pred_images_0[i,pix]=0
                if pred[i,pix,1]>=0.8:
                    pred_images_1[i,pix]=1
                if pred[i,pix,2]>=0.8:
                    pred_images_2[i,pix]=1
                if pred[i,pix,3]>=0.8:
                    pred_images_3[i,pix]=1

    else:
        print "mode " +str(mode) +" not recognized, it can be 'original' or 'threshold'"
        exit()
    pred_images_0 = np.reshape(pred_images_0,(pred_images_0.shape[0],1, patch_height, patch_width))
    pred_images_1 = np.reshape(pred_images_1,(pred_images_1.shape[0],1, patch_height, patch_width))
    pred_images_2 = np.reshape(pred_images_2,(pred_images_2.shape[0],1, patch_height, patch_width))
    pred_images_3 = np.reshape(pred_images_3,(pred_images_3.shape[0],1, patch_height, patch_width))
    return pred_images_0, pred_images_1, pred_images_2, pred_images_3

def combo_to_multiple_imgs(combo_patches, patch_height, patch_width, mode="original"):
    print "shape of combo patches"
    print combo_patches.shape
    print "number of classes in combo images"
    print combo_patches.shape[2]  #check the classes are 4

    class_0_patches = np.empty((combo_patches.shape[0],combo_patches.shape[1])) #(Npatches,height*width)
    class_1_patches = np.empty((combo_patches.shape[0],combo_patches.shape[1])) #(Npatches,height*width)
    class_2_patches = np.empty((combo_patches.shape[0],combo_patches.shape[1])) #(Npatches,height*width)
    class_3_patches = np.empty((combo_patches.shape[0],combo_patches.shape[1])) #(Npatches,height*width)

    if mode=="original":
        for i in range(combo_patches.shape[0]):
            for pix in range(combo_patches.shape[1]):
                class_0_patches[i,pix]=combo_patches[i,pix,0]
                class_1_patches[i,pix]=combo_patches[i,pix,1]
                class_2_patches[i,pix]=combo_patches[i,pix,2]
                class_3_patches[i,pix]=combo_patches[i,pix,3]
    elif mode=="threshold":
        for i in range(combo_patches.shape[0]):
            for pix in range(combo_patches.shape[1]):
                if combo_patches[i,pix,0]==0:
                    class_0_patches[i,pix]=1
                if combo_patches[i,pix,1]==1:
                    class_1_patches[i,pix]=1
                if combo_patches[i,pix,2]==1:
                    class_2_patches[i,pix]=1
                if combo_patches[i,pix,3]==1:
                    class_3_patches[i,pix]=1
        exit()
    class_0_patches = np.reshape(class_0_patches,(class_0_patches.shape[0],1, patch_height, patch_width))
    class_1_patches = np.reshape(class_1_patches,(class_1_patches.shape[0],1, patch_height, patch_width))
    class_2_patches = np.reshape(class_2_patches,(class_2_patches.shape[0],1, patch_height, patch_width))
    class_3_patches = np.reshape(class_3_patches,(class_3_patches.shape[0],1, patch_height, patch_width))
    return class_0_patches, class_1_patches, class_2_patches, class_3_patches

def otsuThreshold(image): 
    thresh = threshold_otsu(image)
    binary = image > thresh
    return binary

#Adds a hard-coded weight value to each class
def addWeightsToPatches(patches_combo_train):
    im_h = patches_combo_train.shape[2]
    im_w = patches_combo_train.shape[3]
    reShapedPatches = np.reshape(patches_combo_train,(patches_combo_train.shape[0],im_h*im_w))
    kerasLabeledPatchesArray = keras.utils.to_categorical(reShapedPatches, num_classes=4)
    reShapedWeights = reShapedPatches

    print "\nNew shape of traning patches:"
    print kerasLabeledPatchesArray.shape

    reShapedWeights[reShapedWeights == 0.0] = 1.0
    reShapedWeights[reShapedWeights == 1.0] = 9.0
    reShapedWeights[reShapedWeights == 2.0] = 1.0
    reShapedWeights[reShapedWeights == 3.0] = 7.0
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
