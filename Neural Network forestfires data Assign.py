#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


forest=pd.read_csv("forestfires.csv")


# In[3]:


forest.head()


# In[4]:


forest.shape


# In[5]:


forest.dtypes


# In[6]:


forest.info()


# In[7]:


forest.loc[forest.size_category=="large","size_category"] = 1
forest.loc[forest.size_category=="small","size_category"] = 0


# In[8]:


forest.size_category.value_counts()


# In[9]:


forest['month']=forest['month'].astype('category')
forest['day']=forest['day'].astype('category')
forest['size_category']=forest['size_category'].astype('int')


# In[10]:


forest.dtypes


# In[11]:


from sklearn import preprocessing                      
label_encoder = preprocessing.LabelEncoder()


# In[12]:


forest['month']=label_encoder.fit_transform(forest['month'])
forest['day']=label_encoder.fit_transform(forest['day'])


# In[13]:


forest


# In[14]:


forest.shape


# In[15]:


X = forest.iloc[:,0:30]
Y = forest.iloc[:,30]


# In[16]:


X


# In[17]:


Y


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=0)


# In[20]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)


# In[21]:


# Now apply the transformations to the data:
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[22]:


from sklearn.neural_network import MLPClassifier


# In[23]:


mlp = MLPClassifier(hidden_layer_sizes=(30,30))


# In[24]:


mlp.fit(x_train,y_train)


# In[25]:


prediction_train=mlp.predict(x_train)
prediction_test = mlp.predict(x_test)


# In[26]:


prediction_test


# In[27]:


prediction_train


# In[28]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,prediction_test))
np.mean(y_test==prediction_test)
np.mean(y_train==prediction_train)


# In[29]:


pd.crosstab(y_test,prediction_test)


# In[ ]:




