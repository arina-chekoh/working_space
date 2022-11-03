#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np

def normalization(data):
    
    
    normalized_dataset = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    """
    norm = np.linalg.norm(data)
    
    matrix = data/norm
    
    Normalize the given data
    Input:  data = a 2d numpy array of shape (n_samples, n_features).
    Output:  the normalized data on each column separately
    """
    return normalized_dataset


# In[4]:


def standardization(data):
    
    standardized_dataset = (data - np.average(data)) / (np.std(data))
    
    """
    standardized_dataset = (dataset - mean(dataset)) / standard_deviation(dataset))
    
    #
    
    Standardize the given data
    Input:  data = a 2d numpy array of shape (n_samples, n_features).
    Output:  the standardized data on each column separately
    """
    return standardized_dataset


# In[7]:


def label_encoding(data):
    
    class_encode = np.unique(data[:], return_inverse=True)
    classes = class_encode[0]
    encoded = class_encode[1]
    
    
    """
    results = np.zeros((len(data), 1))
    for i, label in enumerate(data):
        results[i, data] = 1.
        print(results)
    
    Label encoding the given categorial data in the alphabetical order
    Input:  data = a 1d numpy array of str.
    Output:  the 1d array of encoded labels and a 1d array of class label
    """
    return encoded, classes


# In[8]:


if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, KBinsDiscretizer

    # load iris features and target as numpy array 
    data = np.loadtxt('iris.data', delimiter=',', usecols=(0,1,2,3))
    target = np.loadtxt('iris.data', delimiter=',', usecols=(4), dtype='str')

    scaler = MinMaxScaler()
    skl = scaler.fit_transform(data[:,0:1])
    our = normalization(data[:,0:1])
    print(skl[-5:,:])
    print(our[-5:,:])
    assert np.allclose(skl, our)

    scaler = StandardScaler()
    skl = scaler.fit_transform(data[:,1:3])
    our = standardization(data[:,1:3])
    print(skl[-5:,:])
    print(our[-5:,:])
    assert np.allclose(skl, our)

    le = LabelEncoder()
    skl = le.fit_transform(target)
    our, cls = label_encoding(target)
    print(skl[-5:])
    print(our[-5:])
    print(le.classes_)
    print(cls)


# In[ ]:




