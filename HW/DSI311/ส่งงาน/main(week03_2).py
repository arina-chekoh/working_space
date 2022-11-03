#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math

def one_hot_encoder(target):
    """
    Convert label into one hot encoding format
    Input:  target = 1D array label of the sample
            One-hot encoding format of the target
    """
    target_name, y = np.unique(target, return_inverse=True)
    one_hot = np.zeros((y.size, y.max()+1))
    one_hot[np.arange(y.size),y] = 1
    return target_name, one_hot


# In[2]:


def add_intercept(X):
    """
    Add the intercept (a constant of value 1) to every feature vector
    Input:  X = 2D array of the input data (a row = a sample)
    Output: 2D array of the input data with intercept as an extra column
    """
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate((intercept, X), axis=1)


# In[3]:


def softmax(x):
    
    exp = np.exp(x - np.max(x))
    
    for i in range(len(x)):
        exp[i] /= np.sum(exp[i])
    
    """
    The softmax function takes a vector as input, and normalizes it into a probability distribution.
    Input:  x = input vectors as 2D numpy array (a vector = a row)
    Output: 2D numpy array of probability distribution of each vector from the softmax function.
    """
    return exp


# In[4]:


def loss_func(h, y):
    
    loss = (-y * np.log(h)-(1-y)*np.log(1-h)).mean()
    """
    loss = -np.mean(h*(np.log(y)) - (1-h)*np.log(1-y))
    Log Loss function
    L = - 1/n sum_{i=1}^n y_i log(y'_i)
    Input:  h = 2D array of the class probability estimation for each sample (a row = a sample)
            y = 2D array of the one hot encoding of the label
    """
    return loss


# In[5]:


def find_gradient(X, h, y):
    
    gradient = np.dot(X.T,(h-y)) / y.shape[0]
    """
    Compute gradient
    Input:  X = 2D array of the input data (a row = a sample)
            h = 2D array of the class probability estimation for each sample
            y = 2D array of the one hot encoding of the label
    Output: gradient of the loss function for updating the weights
    
    #grad function
    def grad(x,y,w,):
        grad_w=2 * x * ((w * x) - y ) # derevative with respect to x
        return grad_w
    """
    return gradient


# In[6]:


def fit(X, y, lr, num_iter):
    """
    Training Logistic Regression
    Input:  X = 2D array of the input data (a row = a sample)
            y = 2D array of the one hot encoding of the label
            lr = learning rate
            num_iter = number of the iteration
    Output: the weight of the logistic regression model.
    """
    X = add_intercept(X)
    
    # weights initialization
    theta = np.zeros((X.shape[1], y.shape[1]))

    for i in range(num_iter):
        z = np.dot(X, theta)
        h = softmax(z)
        loss = loss_func(h, y)
        gradient = find_gradient(X, h, y)
        theta -= lr * gradient
            
        if(i % 1000 == 0):
            print(f'Iter {i:4d} \t loss: {loss:.4f} \t')
    print(f'Iter {i:4d} \t loss: {loss:.4f} \t')
    return theta


# In[7]:


def predict_prob(X, theta):
    """
    Make the prediction using Logistic Regression by finding the class probability estimation
    Input:  X = 2D array of the input data (a row = a sample)
            theta = weights of the Logistic Regression model
    Output: the prediction as the class probability estimation
    """
    X = add_intercept(X)
    return softmax(np.dot(X, theta))

def predict(X, theta):
    """
    Make the prediction using Logistic Regression by finding the class label
    Input:  X = 2D array of the input data (a row = a sample)
            theta = weights of the Logistic Regression model
    Output: the prediction as the class label
    """
    return predict_prob(X, theta).argmax(axis=1)


# In[18]:


def find_accuracy(y, prediction):
    
    correct = 0
    total = 0
    for i in range(len(y)):
        act_label = np.argmax(y[i]) # act_label = 1 (index)
        pred_label = np.argmax(prediction[i]) # pred_label = 1 (index)
        if(act_label == pred_label):
            correct += 1
        total += 1
    accuracy = (correct/total)
    """
    Evaluate the accuracy of the prediction
    Input:  y = ground truth in the one-hot encoding format
            prediciton = the prediction as the class label
    Outpu:  accuracy
    
    1.accuracy = (np.sum(y == prediction)) / len(y)
    
    2. 
    num_sample = prediction.shape[0]
    pred_idx = np.argmax(prediction, axis=0)
    acc = np.sum(np.equal(prediction, y).astype(float)) / num_sample
    
    accuracy = (np.sum(y == prediction)) / len(y)
    """
    return accuracy


# In[19]:


import numpy as np 
import math

data = np.loadtxt('iris.data', delimiter=',', usecols=(0,1))
target = np.loadtxt('iris.data', delimiter=',', usecols=(4), dtype='str')

# hyperparameters
lr = 0.1
num_iter=10000

# preprocessing data
X = data
target_name, y = one_hot_encoder(target)
    
# fit Logistic Regression model
theta = fit(X, y, lr, num_iter)
preds = predict(X, theta)

print(f'Accuracy = {find_accuracy(y, preds):.3f}')
print(f'Predictions  = {preds}')
print(f'Groudn Truth = {y.argmax(axis=1)}')


# In[ ]:




