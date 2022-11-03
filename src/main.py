from skimage import io
from matplotlib import pyplot as plt
import numpy as np
from skimage.transform import resize
from multiprocessing import Process

import os
import re

import LBP_processing
import kmeans
import space_transformation
import clustering

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
def main():
    
    im = retrieve_images('melanoma')
    main_task(im, 1, 'melanoma')

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
    Y, gaussian_lbp = LBP_processing.gaussian_lbp(img_array[img_index], r, num)

    # This function returns the LBP treated image in a LAB color space
    lab_image = space_transformation.lab_color_space(Y, gaussian_lbp)

    # This fucntion returns the image after the application of kmeans.
    # It is evident that this is not optimal, we must try to apply the kmeans algorithm only one time.
    clustered = kmeans.kmeans(kmeans.cluster(kmeans.kmeans_colors(lab_image, 2)), 2)

    resizingFactor = 3 # to don't resize, use 1
    clustered = resize(clustered, (clustered.shape[0]//resizingFactor, clustered.shape[1]//resizingFactor))
    plt.imshow(clustered, cmap='gray')
    plt.show()
    clustered = clustering.create_mask(clustered, thresh=0.91, rayon=50//(resizingFactor), show_clustering=1)

    # The line below is used to save the images inside the folder ./out
    io.imsave('out/'+ lesion_type + '/mask_' + id[img_index] + '.png', clustered)
    print(f'Saved image with id: {id[img_index]}')

if __name__ == "__main__":
    main()