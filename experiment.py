#%%
from PIL import Image
from skimage import feature
from matplotlib import pyplot as plt
import numpy as np

def lpb_test(img, r, num):
    img = img.astype(float)
    img = img[:,:,0] * 0.3 + img[:,:,1]*0.59 + img[:,:,2]*0.11

    img = img.astype(np.uint8)

    i_min = np.min(img)
    i_max = np.max(img)

    lbp = feature.local_binary_pattern(img, num, r, method='uniform')

    return lbp



#%%
METHOD = 'uniform'
im = Image.open('src\images\melanoma\ISIC_0000030.jpg')
data = np.asarray(im)

r = 4
num = 20 * r

#%%
lbp = lpb_test(data, r, num)

plt.imshow(lbp/np.max(lbp), cmap='gray')
plt.show()


# %%
