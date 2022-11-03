#!/usr/bin/env python
# coding: utf-8

# In[8]:


n=int(input())
txt=[]
for i in range(n):
    string=input()
    txt.append(string)


# In[9]:


k=int(input())
word=[]
for i in range(k):
    query=input()
    word.append(query)


# In[10]:


for x in word:
    output = []
    result = ''
    for i in range(n):
        if x in txt[i]:
            output.append(str(i+1))
    result = " ".join(output)
    print(result)


# In[ ]:




