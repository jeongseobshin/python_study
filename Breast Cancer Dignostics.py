#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[63]:


import warnings
warnings.filterwarnings('ignore')


# In[64]:


#Load the data
df = pd.read_csv('breastcancer_data.csv')


# In[65]:


df.head()


# In[66]:


df.tail()


# In[67]:


df.shape


# In[68]:


df.describe().T


# In[69]:


df.diagnosis.unique()


# In[70]:


df['diagnosis'].value_counts()


# In[71]:


sns.countplot(df['diagnosis'],palette='husl')


# In[72]:


#Clean and prepare the date

df.drop('id',axis = 1,inplace = True)
df.drop('Unnamed: 32',axis = 1, inplace = True)


# In[44]:


df.head()


# In[45]:


df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
df.head()


# In[46]:


df.isnull().sum()


# In[50]:


#def diagnosis_value(diagnosis):
# if diagnosis == 'M':
#     return 1
# else:
#     return 0
#df['diagnosis'] = df['diagnosis'].apply(diagnosis_value)


# In[48]:


df.corr()


# In[51]:


plt.hist(df['diagnosis'],color='g')
pli.title('plot_Diagnosis (M=1, B=0)')
plt.show()


# In[52]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True)


# In[53]:


cols=['diagnosis',
     'radius_mean',
     'texture_mean',
     'perimeter_mean',
     'area_mean',
     'smoothness_mean',
     'compactness_mean',
     'concavity_mean',
     'concave points_mean',
     'symmetry_mean',
     'fractal_dimension_mean']
sns.pairplot(data=df[cols],hue='diagnosis',palette='rocket')


# almost perfectly linear patterns between the radius, perimeter and area attributes are hinting at the presence of multicollinearity between these variables. (they are highly linearly related) Another set of variables that possibly imply muliticollinearity are the concavity, concave_points and compactness.

# Multicollinearity is a problem as it undermines the significance of independent varibales and we fix it by removing the highly correlated predictors from the model
# Use partial Least Squares Regression (PLS) or Principal Components Analysis, regression methods that cut the number of predictors to a smaller set of uncorrelated components.

# In[55]:


#Generate and cisualize the correlation matrix
corr = df.corr().round(2)

#Mask for the upper truangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

#Set figure size
f, ax = plt.subplots(figsize=(20,20))

#Define. custom colormap
cmap = sns.diverging_paletee(220,10, as_cmap=True)

#Drew the heatmap
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
           square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.tight_layout()


# we can verify the presence of multicollinearity between some of the variables.
# for instance, the radius_mean column has a correlation of 1 and 0.99 with. perimeter_mean and area_mean columns, respectively. This is because the three columns essentially contain the same information, which is the physical size of the observation(the cell).
# Therefore we should only pick ONE of the three columns when we go into futher analysis.

# Another place where multicillinearity is apparent is between the "mean" columns and the "worst" colum.
# For instance, the radius_mean column has a correlation of 0.97 with the radius_worst colum.
# also there is multicollinearity between the attributes compactness, concavity, and concave point. So we can choose just ONE out of these, I am going for Compactness.

# In[57]:


#forst, drop all "worst" columns
cols = ['radius_worst',
       'texture_worst',
       'perimeter_worst',
       'area_worst',
       'smoothness_worst',
       'compactness_worst',
       'concavity_worst',
       'concave points_worst',
       'symmetry_worst',
       'fractal_dimension_worst']
df = df.drop(cols, axis=1)

#then, drop all columns, related to the "perimeter" and "area" attrubutes
cols = ['perimeter_mean',
       'perimeter_se',
       'concave points_mean',
       'concave points_se']
df = df.drop(cols, axis=1)

#verify remaining columns
df.columns


# In[59]:


#Draw the heatmap again, with the new correlation matrix
corr = df.corr().round(2)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize = (20,20))
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
           square=True, linewidths=.5, cbar_kws={"shrink":.5}, annot=True)
plt.tight_layout()


# Building Model

# In[ ]:




