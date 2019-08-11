#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[3]:


df = pd.read_csv('Salary.csv')


# In[4]:


x = df.iloc[:,:-1].values
y = df.iloc[:,1].values


# In[5]:


xTrain ,  xTest , yTrain , yTest = train_test_split(x,y,test_size=1/3, random_state = 0)


# In[6]:


linearRegressor = LinearRegression()


# In[7]:


linearRegressor.fit(xTrain,yTrain)


# In[8]:


yPrediction = linearRegressor.predict(xTest)


# In[17]:


plt.scatter(xTrain, yTrain, color = 'red')
plt.plot(xTrain, linearRegressor.predict(xTrain), color = 'blue')


# In[18]:


plt.scatter(xTest, yTest, color = 'red')
plt.plot(xTrain, linearRegressor.predict(xTrain), color = 'blue')


# In[ ]:




