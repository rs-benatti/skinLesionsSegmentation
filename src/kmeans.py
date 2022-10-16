import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    image = np.zeros((w, h))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

def recreate_image_colorful(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

def kmeans(img, n_class):
    img = np.array(img, dtype=np.float64) / 255
    # Load Image and transform to a 2D numpy array.
    w, h = original_shape = tuple(img.shape)
    d = 1
    image_array = np.reshape(img, (w * h, d))
    image_array_sample = shuffle(image_array, random_state=0)[:100]
    kmeans = KMeans(n_clusters=n_class, random_state=0).fit(image_array_sample)
    labels = kmeans.predict(image_array)
    img = recreate_image(kmeans.cluster_centers_, labels, w, h)
    return img

def kmeans_colors(img, n_colors):
    # Load Image and transform to a 2D numpy array.
    w, h, d = original_shape = tuple(img.shape)
    assert d == 3
    image_array = np.reshape(img, (w * h, d))
    image_array_sample = shuffle(image_array, random_state=0)[:100]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
    labels = kmeans.predict(image_array)
    img = recreate_image_colorful(kmeans.cluster_centers_, labels, w, h)
    return img

def cluster(img):
    shape = (img.shape[0], img.shape[1])
    X = np.array(img[:, :, 0])
    X.reshape(shape)
    X = (X/np.max(X)).astype(int)
    print(X)
    return X
