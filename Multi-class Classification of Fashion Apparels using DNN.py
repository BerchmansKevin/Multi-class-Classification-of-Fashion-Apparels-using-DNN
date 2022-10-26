#!/usr/bin/env python
# coding: utf-8

# ## `Berchmans Kevin S`
# 
# 

# ## `Multi-class Classification of Fashion Apparels using DNN`

# ### 1. Open fashion_mnist dataset from keras

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten


# In[2]:


dataset = tf.keras.datasets.fashion_mnist.load_data()


# ### 2. Perform basic EDA:

# In[3]:


(X_train, y_train), (X_test, y_test) = dataset


# In[4]:


print("X_train shape:", X_train.shape, "           y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape, "            y_test shape:", y_test.shape)


# In[5]:


print("X_train size:", X_train.size, "           y_train size:", y_train.size)
print("X_test size:", X_test.size, "          y_test size:", y_test.size)


# In[6]:


X_train[37]


# In[7]:


y_train[37]


# In[8]:


plt.matshow(X_train[37])
plt.show()


# ### 3. Normalize:

# In[9]:


X_train = X_train.reshape((X_train.shape[0], 28*28)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28*28)).astype('float32')


# In[10]:


X_train = X_train / 255
X_test = X_test / 255


# In[11]:


X_train[37]


# ### 4. Build a simple baseline model:

# In[12]:


model = Sequential()
model.add(Dense(512, input_dim=28*28, activation='relu'))
model.add(Dense(10, activation='softmax'))


# In[13]:


model.compile(loss='mean_squared_error', metrics=['accuracy'])


# In[14]:


model.fit(X_train, y_train, epochs=10)


# In[15]:


model.evaluate(X_test, y_test)


# In[16]:


model.summary()


# ### 5. Performance Analysis:

# 2 layers

# In[17]:


model1 = Sequential()
model1.add(Dense(512, input_dim=28*28, activation='relu'))
model1.add(Dense(512, input_dim=28*28, activation='relu'))
model1.add(Dense(10,activation='softmax'))
model1.compile(loss='mean_squared_error', metrics=['accuracy'])
model1.fit(X_train,y_train,epochs=10)
model1.evaluate(X_test,y_test)


# In[18]:


model2 = Sequential()
model2.add(Dense(256, input_dim=28*28, activation='relu'))
model2.add(Dense(256, input_dim=28*28, activation='relu'))
model2.add(Dense(10,activation='softmax'))
model2.compile(loss='mean_squared_error', metrics=['accuracy'])
model2.fit(X_train,y_train,epochs=10)
model2.evaluate(X_test,y_test)


# In[19]:


model3 = Sequential()
model3.add(Dense(128, input_dim=28*28, activation='relu'))
model3.add(Dense(128, input_dim=28*28, activation='relu'))
model3.add(Dense(10,activation='softmax'))
model3.compile(loss='mean_squared_error', metrics=['accuracy'])
model3.fit(X_train,y_train,epochs=10)
model3.evaluate(X_test,y_test)


# 3 layers

# In[20]:


model4 = Sequential()
model4.add(Dense(512, input_dim=28*28, activation='relu'))
model4.add(Dense(512, input_dim=28*28, activation='relu'))
model4.add(Dense(512, input_dim=28*28, activation='relu'))
model4.add(Dense(10,activation='softmax'))
model4.compile(loss='mean_squared_error', metrics=['accuracy'])
model4.fit(X_train,y_train,epochs=10)
model4.evaluate(X_test,y_test)


# In[21]:


model5 = Sequential()
model5.add(Dense(256, input_dim=28*28, activation='relu'))
model5.add(Dense(256, input_dim=28*28, activation='relu'))
model5.add(Dense(256, input_dim=28*28, activation='relu'))
model5.add(Dense(10,activation='softmax'))
model5.compile(loss='mean_squared_error', metrics=['accuracy'])
model5.fit(X_train,y_train,epochs=10)
model5.evaluate(X_test,y_test)


# In[22]:


model6 = Sequential()
model6.add(Dense(128, input_dim=28*28, activation='relu'))
model6.add(Dense(128, input_dim=28*28, activation='relu'))
model6.add(Dense(128, input_dim=28*28, activation='relu'))
model6.add(Dense(10,activation='softmax'))
model6.compile(loss='mean_squared_error', metrics=['accuracy'])
model6.fit(X_train,y_train,epochs=10)
model6.evaluate(X_test,y_test)


# 4 layers

# In[23]:


model7 = Sequential()
model7.add(Dense(512, input_dim=28*28, activation='relu'))
model7.add(Dense(512, input_dim=28*28, activation='relu'))
model7.add(Dense(512, input_dim=28*28, activation='relu'))
model7.add(Dense(512, input_dim=28*28, activation='relu'))
model7.add(Dense(10,activation='softmax'))
model7.compile(loss='mean_squared_error', metrics=['accuracy'])
model7.fit(X_train,y_train,epochs=10)
model7.evaluate(X_test,y_test)


# In[24]:


model8 = Sequential()
model8.add(Dense(256, input_dim=28*28, activation='relu'))
model8.add(Dense(256, input_dim=28*28, activation='relu'))
model8.add(Dense(256, input_dim=28*28, activation='relu'))
model8.add(Dense(256, input_dim=28*28, activation='relu'))
model8.add(Dense(10,activation='softmax'))
model8.compile(loss='mean_squared_error', metrics=['accuracy'])
model8.fit(X_train,y_train,epochs=10)
model8.evaluate(X_test,y_test)


# In[25]:


model9 = Sequential()
model9.add(Dense(128, input_dim=28*28, activation='relu'))
model9.add(Dense(128, input_dim=28*28, activation='relu'))
model9.add(Dense(128, input_dim=28*28, activation='relu'))
model9.add(Dense(128, input_dim=28*28, activation='relu'))
model9.add(Dense(10,activation='softmax'))
model9.compile(loss='mean_squared_error', metrics=['accuracy'])
model9.fit(X_train,y_train,epochs=10)
model9.evaluate(X_test,y_test)


# 5 layers

# In[26]:


model10 = Sequential()
model10.add(Dense(512, input_dim=28*28, activation='relu'))
model10.add(Dense(512, input_dim=28*28, activation='relu'))
model10.add(Dense(512, input_dim=28*28, activation='relu'))
model10.add(Dense(512, input_dim=28*28, activation='relu'))
model10.add(Dense(512, input_dim=28*28, activation='relu'))
model10.add(Dense(10,activation='softmax'))
model10.compile(loss='mean_squared_error', metrics=['accuracy'])
model10.fit(X_train,y_train,epochs=10)
model10.evaluate(X_test,y_test)


# In[27]:


model11 = Sequential()
model11.add(Dense(256, input_dim=28*28, activation='relu'))
model11.add(Dense(256, input_dim=28*28, activation='relu'))
model11.add(Dense(256, input_dim=28*28, activation='relu'))
model11.add(Dense(256, input_dim=28*28, activation='relu'))
model11.add(Dense(256, input_dim=28*28, activation='relu'))
model11.add(Dense(10,activation='softmax'))
model11.compile(loss='mean_squared_error', metrics=['accuracy'])
model11.fit(X_train,y_train,epochs=10)
model11.evaluate(X_test,y_test)


# In[28]:


model12 = Sequential()
model12.add(Dense(128, input_dim=28*28, activation='relu'))
model12.add(Dense(128, input_dim=28*28, activation='relu'))
model12.add(Dense(128, input_dim=28*28, activation='relu'))
model12.add(Dense(128, input_dim=28*28, activation='relu'))
model12.add(Dense(128, input_dim=28*28, activation='relu'))
model12.add(Dense(10,activation='softmax'))
model12.compile(loss='mean_squared_error', metrics=['accuracy'])
model12.fit(X_train,y_train,epochs=10)
model12.evaluate(X_test,y_test)


# In[ ]:




