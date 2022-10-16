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

def retrieve_images(lesion_type): # 'melanoma' ou 'nevus'
    im = []
    for f in os.listdir('./src/images/' + lesion_type):
        name, extension = os.path.splitext(f)
        try:
            id = str(re.findall("(\d+)$", name)[0])
            im.append(io.imread('src\images\\' + lesion_type + '\ISIC_'+id+'.jpg'))
        except:
            pass
    return im

def retrieve_segmentations(lesion_type):
    im = []
    for f in os.listdir('./src/images/'+ lesion_type):
        name, extension = os.path.splitext(f)
        id = str(re.findall("\d+", name)[0])
        im.append(io.imread('src\images\\' + lesion_type + '\ISIC_'+id+'_Segmentation.jpg'))
    return im

def get_id_list(lesion_type): # 'melanoma' ou 'nevus'
    id_list = []
    for f in os.listdir('./src/images/' + lesion_type):
        name, extension = os.path.splitext(f)
        try:
            id = str(re.findall("(\d+)$", name)[0])
            id_list.append(id)
        except:
            pass

    return id_list

def main():
    im = retrieve_images('melanoma')
    #main_task(im, 2)
    processlist = []
    for img_index in range(0, len(im)):
        processlist.append(Process(target=main_task, args=(im, img_index, 'melanoma')))

    im = retrieve_images('nevus')
    for img_index in range(0, len(im)):
        processlist.append(Process(target=main_task, args=(im, img_index, 'nevus')))

    for process in processlist:
        process.start()

    for process in processlist:
        process.join()

def main_task(img_array, img_index, lesion_type):
    id = get_id_list(lesion_type)
    r = 3
    num = 20 * r
    Y, lbp = LBP_processing.lbp(img_array[img_index], r, num)
    Y, binary_lbp = LBP_processing.binary_lbp(img_array[img_index], r, num)
    Y, gaussian_lbp = LBP_processing.gaussian_lbp(img_array[img_index], r, num)
    lab_image = space_transformation.lab_color_space(Y, gaussian_lbp)
    clustered = kmeans.kmeans(kmeans.cluster(kmeans.kmeans_colors(lab_image, 2)), 2)
    clustered = resize(clustered, (clustered.shape[0] // 3, clustered.shape[1] // 3)) # Aapagr essa linha pra usar imagem inteira
    clustered = clustering.create_mask(clustered, thresh=0.89, rayon=22, x0=clustered.shape[1]//2, y0=clustered.shape[1]//2)
    io.imsave('out/'+ lesion_type + '/mask_' + id[img_index] + '.png', clustered)

if __name__ == "__main__":
    main()