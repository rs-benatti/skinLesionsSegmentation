import cv2
from PIL import Image
from skimage.transform import resize

def hair_removal(im):
    # Original image to grayscale
    resizingFactor = 3 # to don't resize, use 1
    img = cv2.resize(im, dsize=(im.shape[0]//resizingFactor, im.shape[1]//resizingFactor), interpolation=cv2.INTER_CUBIC)

    grayScale = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
    aux = Image.fromarray(grayScale)
    #aux.show()

    # Kernel for the morph operation
    kernel = cv2.getStructuringElement(1,(15,15))

    # Blackhat to find hair contours
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    gaussian= cv2.GaussianBlur(blackhat,(3,3),cv2.BORDER_DEFAULT)

    # intensify the hair countours
    ret,thresh = cv2.threshold(gaussian,10,255,cv2.THRESH_BINARY)

    # inpaint the original image with masks of hairs
    newimg = cv2.inpaint(img,thresh,5,cv2.INPAINT_TELEA)
    aux = Image.fromarray(newimg)
    #aux.show()

    return newimg


def remove_all(im):
    newim = []
    for img in im:
        newim.append(hair_removal(img))
    return newim