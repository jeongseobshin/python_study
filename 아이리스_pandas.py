#!/usr/bin/env python
# coding: utf-8

# In[8]:


import tensorflow as tf
import pandas as pd


# In[9]:


파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv'
아이리스 = pd.read_csv(파일경로)
아이리스.head()


# In[10]:


#원 핫 인코딩
인코딩 = pd.get_dummies(아이리스)
인코딩.head()


# In[11]:


print(인코딩.columns)


# In[12]:


독립 = 인코딩[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
종속 = 인코딩[['품종_setosa', '품종_versicolor', '품종_virginica']]
print(독립.shape , 종속.shape)


# In[14]:


X = tf.keras.layers.Input(shape = [4])
Y = tf.keras.layers.Dense(3, activation = 'softmax')(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics = 'accuracy')


# In[ ]:


model.fit(독립, 종속, epochs=10)


# In[2]:


#모델을 이용합니다
model.predict(독립[0:5])


# In[4]:


print(종속[0:5])


# In[ ]:


#학습한 가중치
model.get_weights()

