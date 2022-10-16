from skimage import io
from matplotlib import pyplot as plt
import numpy as np

import os
import re

import experiment
import LBP_processing
import kmeans
import space_transformation

def retrieve_images():
    im = []
    for f in os.listdir('./src/images/melanoma'):
        name, extension = os.path.splitext(f)
        try:
            id = str(re.findall("(\d+)$", name)[0])
            im.append(io.imread('src\images\melanoma\ISIC_'+id+'.jpg'))
        except:
            pass
    return im

def retrieve_segmentations():
    im = []
    for f in os.listdir('./src/images/melanoma'):
        name, extension = os.path.splitext(f)
        id = str(re.findall("\d+", name)[0])
        im.append(io.imread('src\images\melanoma\ISIC_'+id+'_Segmentation.jpg'))
    return im


def main():
    
    r = 3
    num = 20 * r
    im = retrieve_images()
    Y, lbp = LBP_processing.lbp(im[2], r, num)
    Y, binary_lbp = LBP_processing.binary_lbp(im[2], r, num)
    Y, gaussian_lbp = LBP_processing.gaussian_lbp(im[2], r, num)
    lab_image = space_transformation.lab_color_space(Y, gaussian_lbp)
    clustered = kmeans.kmeans(kmeans.cluster(kmeans.kmeans_colors(lab_image, 2)), 2)
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(im[2])
    ax[1].imshow(lab_image)
    ax[2].imshow(clustered, cmap='gray')
    #ax[1].imshow(lbp/np.max(lbp), cmap='gray')
    plt.show()

if __name__ == "__main__":
    main()