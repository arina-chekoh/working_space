#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import math

def get_affine_transformation_matrix(theta=0, scale=0, translation_x=0, translation_y=0):
    """
    Return Affine Transformation Matrix that performs rotation, scaling, and translation
    Input:  theta = degree of rotation in radians
            scale = scaling ratio
            translation_x = translation in X-axis in pixels
            translation_y = translation in Y-axis in pixels
    Output: the corresponding 2D array affine transformation matrix
    """
    T = np.array([[1,0,translation_x],
                  [0,1,translation_y],
                  [0,0,1]])
    
    R = np.array ([[math.cos(theta),-math.sin(theta),0],
                  [math.sin(theta),math.cos(theta),0],
                  [0,0,1]])
    
    S = np.array([[scale,0,0],
                 [0,scale,0],
                 [0,0,1]])
    
    return T.dot(R.dot(S))


# In[2]:


def get_new_coordinate(T, img):
    """
    Compute the new coordinates of each pixel in img after the transformation T
    Input:  T = trasnformation matrix
            img = image
    Output: coord_out = the new coordinate of of each pixel in img (nx2 array)
    """
    xx, yy = np.meshgrid(range(len(img[0])), range(len(img)))
    coord_in = np.stack((xx.ravel(), yy.ravel(), np.ones((xx.size))))
    coord_out = (T @ coord_in).astype('int')
    
    return coord_out


# In[3]:


def forward_transform(img, d=0, s=1, dx=0, dy=0):
    """
    Perform forward transformation
    Input:  img = image
            d = rotation degree in radian
            s = scaling ratio
            dx = translation in X-axis in pixels
            dy = translation in Y-axis in pixels
    Output: img_transformed = image after forward transformation
    """
    T = get_affine_transformation_matrix(d, s, dx, dy)
    coord_out = get_new_coordinate(T, img)
    img_transformed = np.ones((1500, 1500, 3), dtype=np.uint8)*255
    img_transformed[coord_out[1], coord_out[0], :] = img.reshape((-1,3))
    
    return img_transformed


# In[4]:


def backward_transform(img, d=0, s=1, dx=0, dy=0):
    """
    Perform backward transformation
    Input:  img = image
            d = rotation degree in radian
            s = scaling ratio
            dx = translation in X-axis in pixels
            dy = translation in Y-axis in pixels
    Output: img_transformed = image after backward transformation
    """
    T = get_affine_transformation_matrix(d, s, dx, dy)
    T_inv = np.linalg.inv(T)
    img_transformed = np.ones((1500, 1500, 3), dtype=np.uint8)*255
    coord_inv = get_new_coordinate(T_inv, img_transformed)
    coord_inv_bound = np.around(coord_inv)
    idx = (coord_inv_bound[1] >= 0) & (coord_inv_bound[1] < len(img)) & (coord_inv_bound[0] >= 0) & (coord_inv_bound[0] < len(img[0]))
    idx2d = idx.reshape(img_transformed.shape[:-1])
    img_transformed[idx2d,:] = img[coord_inv_bound[1, idx], coord_inv_bound[0, idx], :]
   
    return img_transformed


# In[5]:


if __name__=="__main__":
    img = plt.imread('cat.jpg')
    img_forward_transformed = forward_transform(img, d=-np.pi/4, s=2, dy=600, dx=100)
    img_backward_transformed = backward_transform(img, d=-np.pi/4, s=2, dy=600, dx=100)

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img_forward_transformed)
    plt.title('forward warping')
    plt.subplot(1,2,2)
    plt.imshow(img_backward_transformed)
    plt.title('backward warping')
    plt.show()


# In[ ]:




