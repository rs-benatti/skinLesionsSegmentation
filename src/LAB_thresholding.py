import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import LBP_processing
import space_transformation

def lab_cumulative_histogram(im, plotFlag = 1):
    im = im[:, :, 0]
    histo, bin_edges = np.histogram(im, bins=256, range=(0, np.max(im)))
    histo=histo/histo.sum()
    histocum=histo.cumsum()
    if (plotFlag):
        plt.figure()
        plt.title("Histogram")
        plt.xlabel("Value")
        plt.ylabel("pixel count")
        plt.xlim([0.0, np.max(im)])
        plt.plot(bin_edges[0:-1], histocum)
        plt.show()
    return (histocum, bin_edges[0:-1])

# Returns the limit value for a percentage of the histogram
# The pixels within a value below the returned are the darkers
def get_dark_pixels(histo, bins, percentage):
    histo = histo[histo < percentage]
    return bins[len(histo)]

def get_locations_of_pixels_below_x(im, x):
    im = im[:, :, 0]
    return np.nonzero(im < x)

def get_distance_from_center(im, line, column):
    center = (im.shape[0] // 2, im.shape[1] // 2)
    distances = np.zeros(len(line))
    for i in range(0, len(line)):
        distance = ((line[i] - center[0])**2 + (column[i] - center[1])**2)**0.5
        distances[i] = distance
        
    return distances

def get_most_centralized_pixel_value(im, line, column):
    distances = get_distance_from_center(im, line, column)
    index = np.argmin(distances)
    return im[line[index], column[index]]

def get_l_threshold(im, percentage):
    (histocum, bins) = lab_cumulative_histogram(im, plotFlag = 0)
    limit_value = get_dark_pixels(histocum, bins, percentage)
    locations = get_locations_of_pixels_below_x(im, limit_value)
    line = locations[0]
    column = locations[1]
    return limit_value
    return get_most_centralized_pixel_value(im, line, column)[0]

def main():
    im = io.imread('src\images\\melanoma\ISIC_0000049.jpg')
    # This implementation returns the Y scale of the original image without the application of a gaussian filter and
    # the image itself after a LBP, binarization and application of a gaussian filter
    Y, gaussian_lbp = LBP_processing.gaussian_lbp(im, 3, 60)
    # This function returns the LBP treated image in a LAB color space
    im= space_transformation.lab_color_space(Y, gaussian_lbp)
    # create the histogram
    threshold = get_l_threshold(im, 0.99)
    print(threshold)
    binary = im[:, :, 0] > threshold
    plt.imshow(binary)
    plt.show()

if __name__ == "__main__":
    main()