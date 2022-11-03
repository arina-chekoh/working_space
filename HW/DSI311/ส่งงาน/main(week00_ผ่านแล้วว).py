#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
n_equ = int(input())
A = []
B = []
for i in range(n_equ) :
    V = list(map(float, input().split()))
    for n in range(len(V)) :
        if n != n_equ : 
            A.append(V[n])  
        else :
            B.append(V[n])
        i+1
A = np.array(A).reshape(-1,n_equ)
B = np.array(B)
 
X = np.linalg.solve(A,B)
for i in range(len(X)):
    print('{:.2f}'.format(X[i]))
    i+1

