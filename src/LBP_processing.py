import numpy as np
from skimage import feature
from skimage import filters

def lbp(img, r, num):
    # Obs.: Essa função tá retornando valores invertidos em relação ao esperado pela descrição dada nos artigos
    # De acordo com os artigos, dentro da lesão deveríamos ter mais 1s e fora mais 0s
    img = np.asarray(img)
    img = img.astype(float)
    #img = img[:,:,0] * 0.3 + img[:,:,1]*0.59 + img[:,:,2]*0.11
    # Ajustando pra valores da referência [6] do paper:
    img = img[:, :, 0] * 0.2989 + img[:, :, 1] * 0.5870 + img[:, :, 2] * 0.1140

    img = img.astype(np.uint8)

    i_min = np.min(img)
    i_max = np.max(img)

    lbp = feature.local_binary_pattern(img, num, r, method='uniform') # Falta checar se esse é o método que a gnt quer mesmo, o artigo fala um pouco sobre rotação

    return img, lbp

def binary_lbp(img, r, num):
    Y, lbp_temp = lbp(img, r, num)
    return Y, np.around(lbp_temp/np.max(lbp_temp))

def gaussian_lbp(img, r, num):
    Y, lbp = binary_lbp(img, r, num)
    return Y, (filters.gaussian(lbp, sigma=3) * 255)