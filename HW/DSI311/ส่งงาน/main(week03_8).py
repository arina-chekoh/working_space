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


# In[22]:


def softmax(x):
    
    Ex = np.exp(x)
    S = Ex / Ex.sum(axis=1,keepdims = True)
    
    """
    The softmax function takes a vector as input, and normalizes it into a probability distribution.
    Input:  x = input vectors as 2D numpy array (a vector = a row)
    Output: 2D numpy array of probability distribution of each vector from the softmax function.
    
    1.exp = np.exp(x - np.max(x))
    
    for i in range(len(x)):
        exp[i] /= np.sum(exp[i])
    
    
    """
    return S


# In[37]:


def loss_func(h, y):
    
    h = np.clip(h, 1e-10, 1 - 1e-10)

    if h.shape[1] == 1:
        h = np.append(1 - h, h, axis=1)

    if y.shape[1] == 1:
        y = np.append(1 - y, y, axis=1) 
    
    log_loss = -np.sum(y * np.log(h)) / h.shape[0]
    
    """
    loss = -np.mean(h*(np.log(y)) - (1-h)*np.log(1-y))
    Log Loss function
    L = - 1/n sum_{i=1}^n y_i log(y'_i)
    Input:  h = 2D array of the class probability estimation for each sample (a row = a sample)
            y = 2D array of the one hot encoding of the label
            
    1.loss = (-y * np.log(h)-(1-y)*np.log(1-h)).mean()
    """
    return log_loss


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


# In[88]:


def find_accuracy(y, prediction):
    
    num_sample = prediction.shape[0]
    pred_idx = np.argmax(prediction, axis=0)
    acc = (np.sum(np.equal(y.reshape(3,150), prediction).astype(float))) / num_sample
        
    
    
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
    
    3.correct = 0
    total = 0
    for i in range(len(y)):
        act_label = np.argmax(y[i]) # act_label = 1 (index)
        pred_label = np.argmax(prediction[i]) # pred_label = 1 (index)
        if(act_label == pred_label):
            correct += 1
        total += 1
    accuracy = (correct / total)
    
    4.correct = 0
    total = prediction.shape[0] * prediction.shape[1]

    for i, pred_row in enumerate(prediction):
        for j, per_pred in enumerate(pred_row):
            if per_pred >= 0.5 and y[i][j] >= 0.5:
                correct += 1
    acc = float(correct * 100) / total
    
    5.correct = 0
    length = len(y)
    for i in range(length) :
        prediction = round(y[i])
        ans = y[i]
    if prediction == ans.all():
        correct += 1
    
    accuracy = (correct / length)
    """
    
    
    
    return acc


# In[89]:


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




