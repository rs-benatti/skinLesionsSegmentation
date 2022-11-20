import numpy as np
import matplotlib.pyplot as plt
paddling = 10
def get_last_dark_pixel(img):
     # Obs.: Essa função tá retornando valores invertidos em relação ao esperado pela descrição dada nos artigos
    # De acordo com os artigos, dentro da lesão deveríamos ter mais 1s e fora mais 0s
    img = np.asarray(img)
    img = img.astype(float)
    #img = img[:,:,0] * 0.3 + img[:,:,1]*0.59 + img[:,:,2]*0.11
    # Ajustando pra valores da referência [6] do paper:
    img = img[:, :, 0] * 0.2989 + img[:, :, 1] * 0.5870 + img[:, :, 2] * 0.1140
    
    largest_i = -1
    threshold = 100
    for i in range(0, int(len(img[paddling, :])/2)):
        pixel = img[paddling, i]
        if pixel < threshold :
            largest_i = i

    largest_j = -1
    for j in range(0, int(len(img[:, paddling])/2)):
        pixel = img[j, paddling]
        if pixel < threshold:
            largest_j = j
        
    return (int(0.1*largest_j), int(0.8*largest_i))

def crop_image(im, corner):
    # (x0, y0)            (x0, y1)
    #       **************
    #       *            *       
    #       *            *       
    #       *            *       
    #       *            *       
    #       *            *
    #       *            *
    #       **************
    # (x1, y0)            (x1, y1)

    # Some paddling:
    if corner[0] < paddling:
        x0 = paddling * 2
        y0 = paddling
        x1 = len(im[:, 0]) - x0
        y1 = len(im[0, :]) - y0
    else:
        x0 = corner[0]
        y0 = corner[1] 
        x1 = len(im[:, 0]) - x0
        y1 = len(im[0, :]) - y0
    im = im[x0:x1, y0:y1]
    return im

def retrieve_no_corner_images(img):
    cropped_images = []
    for im in img:
        corner = get_last_dark_pixel(im)
        im = crop_image(im, corner)
        cropped_images.append(im)
    return cropped_images
