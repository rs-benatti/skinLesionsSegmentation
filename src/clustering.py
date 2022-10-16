import skimage.morphology as morpho  
import numpy as np
import matplotlib.pyplot as plt
import cv2

def my_perimeter(im):
    se=morpho.disk(1)
    dil=morpho.dilation(im,se)
    diff=dil-im
    ta=np.nonzero(diff)
    return ta

def create_mask(im, thresh=0.85, rayon=9, x0=300, y0=300):
    mask=np.zeros((len(im),len(im[0])))
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
    return mask

def plot_superposition(im, mask):
    plt.figure('superposition')
    plt.imshow(im, cmap='gray') # I would add interpolation='none'
    plt.imshow(mask, cmap='jet', alpha=0.5) # interpolation='none'
    plt.show()