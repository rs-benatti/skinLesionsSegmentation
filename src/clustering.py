import skimage.morphology as morpho  
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

def my_perimeter(im):
    se=morpho.disk(1)
    dil=morpho.dilation(im,se)
    diff=dil-im
    ta=np.nonzero(diff)
    return ta

def create_mask(im, thresh=0.85, rayon=9, show_clustering=0):
    m0 = im[:, :].max()
    iterCounter = 0
    rayon_temp = rayon // 4
    while ((m0 > (0.4 * im[:, :].max()) or s0 == 0) or np.isnan(m0) == True):
        center = get_center_pixel(im)
        x0 = center[0]
        y0 = center[1]
        mask=np.zeros((len(im[0]),len(im[1])))
        mask[y0,x0]=255
        ext=im[y0-rayon_temp:y0+rayon_temp+1,x0-rayon_temp:x0+rayon_temp+1]
        m0=np.mean(ext)
        s0=np.std(ext)
        if iterCounter > 1000:
            print("Reached stop condition")
            x0 = im.shape[1] // 2
            y0 = im.shape[0] // 2
            mask=np.zeros((len(im[0]),len(im[1])))
            mask[y0,x0]=255
            ext=im[y0-rayon_temp:y0+rayon_temp+1,x0-rayon_temp:x0+rayon_temp+1]
            m0=np.mean(ext)
            s0=np.std(ext)
            print(m0)
            print(s0)
            break
        print(m0)
        print(s0)
        iterCounter += 1
    mask=np.zeros((len(im[0]),len(im[1])))
    mask[y0,x0]=255
    ext=im[y0-rayon:y0+rayon+1,x0-rayon:x0+rayon+1]
    m0=np.mean(ext)
    s0=np.std(ext)
    modif=1
    iter = 0 
    while modif >  0:
        iter=iter+1
        modif=0
        per=my_perimeter(mask)
        for i in range (0 , len(per[0])):
            y=per[0][i]
            x=per[1][i]
            ext=im[y-rayon:y+rayon+1,x-rayon:x+rayon+1]
            m=np.mean(ext)
            s=np.std(ext)
            if np.abs(m0-m) < thresh * s0 :
                mask[y][x]=255
                modif=1
        if(show_clustering):
            cv2.imshow('frame',mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    return mask

def plot_superposition(im, mask):
    plt.figure('superposition')
    plt.imshow(im, cmap='gray') # I would add interpolation='none'
    plt.imshow(mask, cmap='jet', alpha=0.5) # interpolation='none'
    plt.show()

# Check if the pixel in the center of the image is part of the lesion (wether it's black)
# If it's not, searchs for a black pixel in the image
# Returns a tuple with the  pixel coordinates
def get_center_pixel(im):
    print("Called function get_center_pixel")
    x0 = im.shape[0]//2
    y0 = im.shape[1]//2
    center=(x0, y0)
    centerPixel  = im[x0][y0]
    threshold = 0.8 * im[:, :].max()
    minimum = im[:, :].min()
    side = int(0.4 * min([im.shape[0], im.shape[1]]))
    print(im.shape)
    while (centerPixel != minimum):
        x0 = random.randint(0, im.shape[0] - 1)
        y0 = random.randint(0, im.shape[1] - 1)
        centerPixel  = im[x0][y0]
        center=(x0, y0)
    return center
