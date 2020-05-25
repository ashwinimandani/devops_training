#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd


# In[40]:


dataset = pd.read_csv('weight-height.csv')


# In[41]:


dataset.info()


# In[42]:


dataset.columns


# In[43]:


y = dataset['Weight']


# In[44]:


X = dataset['Height']


# In[45]:


from keras.models import Sequential


# In[46]:


from keras.layers import Dense


# In[47]:


from keras.optimizers import Adam


# In[48]:


model = Sequential()


# In[49]:


# by default : linear activation functions
# units : output
# input shape: input feature
# dense: how many hidden layer
model.add(Dense(units=1 , input_shape=(1,)  ))


# In[50]:


model.summary()


# In[51]:


model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.000001) )


# In[74]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y)
model.fit(x_train,y_train,epochs=100)


# In[75]:


y_pred=model.predict(x_test)
list1=[]


# In[76]:


# for i in range(len(y_pred)):
#     list1.append(abs(y_pred[i]-y_test[i]))
y_pred[5]


# In[77]:


for i in range(len(y_pred)):
    list1.append((abs(y_pred[i]-y_test.values[i])/y_test.values[i])*100)


# # 

# In[80]:


y_pred[1]


# In[72]:


y_test.values


# In[ ]:




