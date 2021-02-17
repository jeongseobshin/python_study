#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd


# In[2]:


path = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
boston = pd.read_csv(path)

print(boston.columns)
boston.head()


독립 = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']]
종속 = boston[['medv']]
print(독립.shape ,종속.shape)


# In[14]:


X = tf.keras.layers.Input(shape=[13])
H = tf.keras.layers.Dense(10, activation='swish')(X)
Y = tf.keras.layers.Dense(1)(H)
model = tf.keras.models.Model(X,Y)
model.compile(loss='mse')


# In[11]:


model.summary()


# In[9]:


model.fit(독립, 종속, epochs = 10)


# In[10]:


model.predict(독립[0:5])

종속[0:5]

