import matplotlib.pyplot as plt
import numpy as np


def rgb2gray(rgb):
    """
    Convert image from RGB into grayscale
    Input:  rgb = RGB image
    Output: grayscale image
    """
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def partial_derivative(gray_image):
    """
    Find partial derivative using [-1 0 1] and [-1 0 1]T as x-derivative and y-derivate respectively.
    Input:  gray_image = grayscale image
    Output  Ix = x-derivative of image
            Iy = y-derivative of image
    """
    padded_gray_image = np.pad(gray_image, 1, mode='edge')
    Iy = padded_gray_image[2:, 1:-1] - padded_gray_image[:-2, 1:-1]
    Ix = padded_gray_image[1:-1, 2:] - padded_gray_image[1:-1, :-2]
    return Ix, Iy

def findCorners(image, window_size, k=0.04):
    """
    Compute the Harris Corner Detector score
    Input:  image = grayscale image
            window_size = the size (side length) of the sliding window (must be odd value)
            k = Harris corner constant. Usually 0.04 - 0.06
    Output: the Harris Corner Detector score of every pixel as the 2D numpy array
    :return:
    """
    # find second moment
    padded_size = window_size//2
    padded_image = np.pad(image, padded_size, 'edge')
    Ix, Iy = partial_derivative(padded_image)
    Ixx = Ix**2
    Ixy = Iy*Ix
    Iyy = Iy**2
    
    height, width = image.shape
    output = np.zeros(image.shape)
    # Hint: Loop through every pixel in the image and compute score of being corner

    return output

def nms(output, window_size):
    """
    Perform non-maximum suppression to find the coordinates of the corners
    Input:  output = the Harris Corner Detector score of every pixel as the 2D numpy array
            window_size = the size (side length) of the sliding window (must be odd value)
    Output: coordinates of the corner points
    """    
    threshold = 0.01*output.max()
    height, width = output.shape
    corners = np.zeros(output.shape, dtype='bool')
    padded_size = window_size//2
    padded_output = np.pad(output, padded_size, 'edge')
    for y in range(height):
        for x in range(width):
            #Calculate sum of squares
            window = padded_output[y:y+window_size, x:x+window_size]
            
            if output[y,x] > threshold and output[y,x] == window.max():
                corners[y,x] = 1 # maxmimum value
                print(window, output[y,x])
    return np.where(corners)

if __name__ == "__main__":
    img = plt.imread('chess_board.jpg')
    img = img / 255.

    window_size = 11
    k = 0.04
    
    gray_img = rgb2gray(img)
    output = findCorners(gray_img, window_size, k)

    corners_points = nms(output, window_size)
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.title('original image')
    plt.subplot(1,3,2)
    plt.imshow(output)
    plt.title('Harris response')
    plt.subplot(1,3,3)
    plt.imshow(gray_img, cmap='gray')
    plt.scatter(corners_points[1], corners_points[0], s=12, c='r', marker='x')
    plt.title('Corners')
    plt.show()
    