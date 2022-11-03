#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np

def gaussian_kernel(size, sigma=1):
    """
    Generate 2D Gaussian kernel
    Input:  size = size of the Gaussian kernel
            sigma = sigma of the Gaussian function
    Output: 2D array of the Gaussian kernel
    """
    n = size//2
    xx, yy = np.meshgrid(range(-n,n+1), range(-n,n+1))
    kernel = np.exp(- (xx**2 + yy**2) / (2*sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


# In[2]:


def gaussian_blur(image, kernel_size=5, sigma=1):
    """
    Perform Gaussian blur on the image
    Input:  image = the original image to perform Gaussian blur on
            kernel_size = size of the Gaussian kernel
            sigma = sigma of the Gaussian function
    Output  image after applied the Gaussian blue
    """
    kernel = gaussian_kernel(kernel_size, sigma)

    pad_height = int((kernel_size-1)/2)
    pad_width = int((kernel_size-1)/2)

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), 'edge')
    
    image_row, image_col, image_color = image.shape
    kernel_row, kernel_col = kernel.shape
    output = np.zeros(image.shape)

    for row in range(image_row):
        for col in range(image_col):
            for color in range(image_color):
                collector = []
                for k_row in range(kernel_row):
                    for k_col in range(kernel_col):
                        collector.append(kernel[k_row, k_col]*padded_image[row+k_row, col+k_col][color])
                output[row, col][color] = sum(collector)/np.sum(kernel)
    
    return output


# In[3]:


if __name__ == "__main__":
    img = plt.imread('cat.jpg')
    img = img/255.
    img_blur = gaussian_blur(img)

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title('original')
    plt.subplot(1,2,2)
    plt.imshow(img_blur)
    plt.title('blurred image')
    plt.show()


# In[ ]:




