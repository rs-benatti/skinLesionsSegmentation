import os
import re
from multiprocessing import Process
import scipy
import cv2

import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage.transform import resize

import clustering
import corner_removal
import kmeans
import LAB_thresholding
import LBP_processing
import space_transformation
import clustering
import Hair_removal
import CCL

# Obs.: corner_removal is not perfect yet, it doesn't work with lesions in skins with a lot of hair.
# This only happens in nevus images, after adding the hair removal we will need to test it again


# This function returns a list of io images from the group passed as parameter ('melanoma' or 'nevus')
def retrieve_images(lesion_type): # Parameter: 'melanoma' or 'nevus'
    im = []
    for f in os.listdir('./src/images/' + lesion_type):
        name, extension = os.path.splitext(f)
        try:
            id = str(re.findall("(\d+)$", name)[0])
            image = io.imread('src\images\\' + lesion_type + '\ISIC_'+id+'.jpg')
            im.append(image)
        except:
            pass
    return im

# This function returns an array of segmented io images from the group passed as parameter ('melanoma' or 'nevus')
# This function is not being used in code yet, it's going to be used further
def retrieve_segmentations(lesion_type):
    im = []
    for f in os.listdir('./src/images/'+ lesion_type):
        name, extension = os.path.splitext(f)
        id = str(re.findall("\d+", name)[0])
        im.append(io.imread('src\images\\' + lesion_type + '\ISIC_'+id+'_Segmentation.jpg'))
    return im

# This function returns a list of the id of the images gro the group passed as parameter
def get_id_list(lesion_type): # 'melanoma' or 'nevus'
    id_list = []
    for f in os.listdir('./src/images/' + lesion_type):
        name, extension = os.path.splitext(f)
        try:
            id = str(re.findall("(\d+)$", name)[0])
            id_list.append(id)
        except:
            pass

    return id_list


# This is the main function.
plot = 0
def main():
    type = 'melanoma'
    im = retrieve_images(type)
    im = Hair_removal.remove_all(im)
    im = corner_removal.retrieve_no_corner_images(im)
    plt.imshow(im[9])
    plt.show()
    #main_task(im, 9, 'melanoma')
    main_task(im, 9, type)
    
    for index in range(0, len(im)):
        main_task(im, index, type)
    

    '''
    processlist = []

    # The code block below serves only to append the parallel processes
    for img_index in range(0, len(im)):
        processlist.append(Process(target=main_task, args=(im, img_index, 'melanoma')))
    im = retrieve_images('nevus')
    for img_index in range(0, len(im)):
        processlist.append(Process(target=main_task, args=(im, img_index, 'nevus')))
    for process in processlist:
        process.start()
    for process in processlist:
        process.join()
    '''
# This code is currently using parallel processes, so, the main function calls the function below.
def main_task(img_array, img_index, lesion_type):
    id = get_id_list(lesion_type)
    r = 3
    num = 20 * r

    # This implementation returns the Y scale of the original image without the application of a gaussian filter and
    # the image itself after a LBP, binarization and application of a gaussian filter
    im = img_array[img_index]
    Y, gaussian_lbp = LBP_processing.gaussian_lbp(im, r, num)

    # This function returns the LBP treated image in a LAB color space
    lab_image = space_transformation.lab_color_space(Y, gaussian_lbp)

    '''
    # This fucntion returns the image after the application of kmeans.
    # It is evident that this is not optimal, we must try to apply the kmeans algorithm only one time.
    clustered = kmeans.kmeans(kmeans.cluster(kmeans.kmeans_colors(lab_image, 2)), 2)
    '''

    #The application of kmeans is substituted by a thresholding method

    threshold = LAB_thresholding.get_l_threshold(lab_image, 0.45) # 0.5
    binary = lab_image[:, :, 0] > threshold
    if (lesion_type == 'nevus' and img_index != 4):
        binary = kmeans.kmeans(kmeans.cluster(kmeans.kmeans_colors(lab_image, 2)), 2)
    resizingFactor = 1 # to don't resize, use 1
    binary = resize(binary, (binary.shape[0]//resizingFactor, binary.shape[1]//resizingFactor))
    #binary = CCL.CCL(binary)
    if plot == 1:
        plt.imshow(binary, cmap='gray')
        plt.show()
    #plt.show()
   # clustered = clustering.create_mask(binary, thresh=1, rayon=200//(resizingFactor), show_clustering=1) 
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)) # (10, 10) for melanoma
    
    clustered = cv2.morphologyEx(binary, cv2.MORPH_OPEN, opening_kernel)
    #clustered = 1-scipy.ndimage.binary_fill_holes(clustered)
    clustered = 1-clustered
    #opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10)) # (7, 7)
    clustered = clustered.astype('uint8')
    #clustered = cv2.morphologyEx(clustered, cv2.MORPH_OPEN, opening_kernel)
    if plot == 1:
        plt.imshow(clustered, cmap='gray')
        plt.title("Before closing")
        plt.show()
    #plt.show()
    clustered = cv2.morphologyEx(clustered, cv2.MORPH_CLOSE, kernel)
    
    if plot == 1:
        plt.imshow(clustered, cmap='gray')
        plt.title("After closing")
        plt.show()
    #plt.show()
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40,40))
    clustered = cv2.morphologyEx(clustered, cv2.MORPH_OPEN, opening_kernel)
    #clustered = CCL.CCL(clustered)
    if plot == 1:
        plt.imshow(clustered, cmap='gray')
        plt.show()
    plt.imsave('out/'+ lesion_type + '/mask_' + id[img_index] + '.png', clustered, cmap='gray')
    print(f'Saved image with id: {id[img_index]}')
    #plt.show()
    
    im = resize(im, (im.shape[0]//resizingFactor, im.shape[1]//resizingFactor))
    #plt.figure('superposition')
    
    plt.imshow(im, cmap='gray') # I would add interpolation='none'
    plt.imshow(clustered, cmap='jet', alpha=0.5) # interpolation='none'
    plt.savefig('out/'+ lesion_type + '/mask_' + id[img_index] + '_superposition.png')
    if plot == 1:
        plt.show()
    #plt.show()
    # The line below is used to save the images inside the folder ./out
    #io.imsave('out/'+ lesion_type + '/mask_' + id[img_index] + '.png', clustered)
    #print(f'Saved image with id: {id[img_index]}')

if __name__ == "__main__":
    main()
    '''
    img = retrieve_images('melanoma')
    img = corner_removal.retrieve_no_corner_images(img)
    for im in img:
        plt.imshow(im)
        plt.show()
    '''
    