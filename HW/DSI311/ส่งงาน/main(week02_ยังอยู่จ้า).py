#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np

def find_accuracy(y_true, y_pred):
    
    accuracy = len([y_true[i] for i in range(0, len(y_true)) if y_true[i] == y_pred[i]]) / len(y_true)
    
    "accuracy = np.sum(np.equal(y_true, y_pred)) / len(y_true)"
    
     
    return accuracy


# In[20]:


def find_precision(y_true, y_pred):
    
   
    tp = []
    tn = []
    fp = []
    fn = []
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            if y_true[i] == 1:
                tp.append(1)
            else : 
                tn.append(1)

        else : 
            if y_pred[i] == 1 :
                fp.append(1)
            else :
                fn.append(1)
                   
    precision = len(tp) / (len(tp + fp))
                   
    return precision


# In[16]:


def find_recall(y_true, y_pred):
    
    tp = []
    tn = []
    fp = []
    fn = []
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            if y_true[i] == 1:
                tp.append(1)
            else : 
                tn.append(1)

        else : 
            if y_pred[i] == 1 :
                fp.append(1)
            else :
                fn.append(1)
                   
    recall = len(tp) / (len(tp + fn))
    
    return recall


# In[14]:


def find_f1score(y_true, y_pred):
    
    precision = find_precision(y_true, y_pred)
    recall = find_recall(y_true, y_pred)
    
    f1_score = 2 * ((precision * recall) / (precision + recall))
    
    
    return f1_score


# In[21]:


if __name__ == "__main__":
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

    # load iris features and target as numpy array 
    data = np.loadtxt('iris.data', delimiter=',', usecols=(0,1))
    target = np.loadtxt('iris.data', delimiter=',', usecols=(4), dtype='str')

    # convert target (in str) to number (int)
    le = LabelEncoder()
    y = le.fit_transform(target)
    print(y, type(y.dtype), le.classes_)

    # split data to create training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

    # build logistic regression model and make prediction
    clf = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(y_pred, y_test)

    # evaluate 
    print(f'Accuracy = {accuracy_score(y_test, y_pred):.3f}')
    print(f'Accuracy = {find_accuracy(y_test, y_pred):.3f}')
    print(f'Precision = \n{precision_score(y_test, y_pred, average=None)}')
    print(f'Precision = \n{find_precision(y_test, y_pred):.3f}')
    print(f'Recall = \n{recall_score(y_test, y_pred, average=None)}')
    print(f'Recall = \n{find_recall(y_test, y_pred):.3f}')
    print(f'F1-score = \n{f1_score(y_test, y_pred, average=None)}')
    print(f'F1-score = \n{find_f1score(y_test, y_pred):.3f}')


# In[ ]:




