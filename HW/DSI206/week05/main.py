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


# In[4]:


def gaussian_blur_fft(image, kernel_size=5, sigma=1):
    """
    Perform Gaussian blur on the image using FFT
    Input:  image = the original image to perform Gaussian blur on
            kernel_size = size of the Gaussian kernel
            sigma = sigma of the Gaussian function
    Output: Image after applied the Gaussian blur using FFT
            Gaussian kernel in the frequency domain
            Image in the frequency domain
            Convolved image in the frequency domain
    """
    kernel = gaussian_kernel(kernel_size, sigma)
 
    pad_height = int((kernel_size-1)/2)
    pad_width = int((kernel_size-1)/2)

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), 'edge')

    kernel_fft = np.fft.fft2(kernel, s=padded_image.shape[:2], axes=(0, 1))

    image_fft = np.fft.fft2(padded_image, axes=(0, 1))

    convolved_fft = kernel_fft[:, :, np.newaxis] * image_fft
    
    inverse_fft = np.fft.ifft2(convolved_fft, axes=(0, 1))
    img_blur = np.real(inverse_fft)
    crop_img = img_blur[2*pad_height:len(inverse_fft), 2*pad_width:len(inverse_fft[0]), :]

    return crop_img, kernel_fft, image_fft, convolved_fft


# In[5]:


if __name__ == "__main__":
    img = plt.imread('cat.jpg')
    img = img/255.
    img_blur, kernel_fft, image_fft, convolved_fft = gaussian_blur_fft(img)

    image_fft_shift = np.log( abs(np.fft.fftshift(image_fft).real) + 1 )
    kernel_fft_shift = np.fft.fftshift(kernel_fft)
    convolved_fft_shift = np.log( abs(np.fft.fftshift(convolved_fft).real) + 1 )

    plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(img)
    plt.title('original')
    plt.subplot(2,3,2)
    plt.imshow(img_blur)
    plt.title('blurred image')
    
    plt.subplot(2,3,4)
    plt.imshow(kernel_fft.real)
    plt.title('kernel')
    plt.subplot(2,3,5)
    plt.imshow(kernel_fft_shift.real)
    plt.title('shifted kernel')
    plt.subplot(2,3,6)
    plt.imshow(image_fft_shift / image_fft_shift.max())
    plt.title('shifted image fft magnitude')
    plt.subplot(2,3,3)
    plt.imshow(convolved_fft_shift / convolved_fft_shift.max())
    plt.title('convolved image fft magnitude')
    plt.show()


# In[ ]:




