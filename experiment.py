#%%
from PIL import Image
from skimage import feature
from skimage import data
from matplotlib import pyplot as plt
import numpy as np

def lpb_test(img, r, num):

    img = img.astype(np.uint8)

    i_min = np.min(img)
    i_max = np.max(img)

    lbp = feature.local_binary_pattern(img, num, r, method='uniform')

    return lbp



#%%
METHOD = 'uniform'
im = data.camera()
dado = np.asarray(im)

r = 0.5
num = 5 * r

#%%
lbp = lpb_test(dado, r, num)

plt.imshow(lbp/np.max(lbp), cmap='gray')
plt.show()


# %%
