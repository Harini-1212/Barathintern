#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


df=pd.read_csv('EW-MAX.xls')


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


df.info()


# In[ ]:


sorted_df = df.sort_values('Volume',ascending=True).head(10)
sorted_df


# In[ ]:


sorted_df = df.sort_values("High",ascending=True).head(10)
sorted_df


# In[ ]:


df.describe().transpose


# In[ ]:


most_volume = df.query('Volume>1500000', inplace = False).sort_values('Volume', ascending = False)
most_volume[:10]


# In[ ]:


df[["High"]].iloc[18]


# In[ ]:


df[["Low"]].iloc[10]


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


sns.countplot(df.Volume)
df.xlabel='Volume'
df.ylabel='count'


# In[ ]:


sns.histplot(df.High)


# In[ ]:


sns.boxplot(df.Low)


# In[ ]:


sns.violinplot(df.Volume)


# In[ ]:


plt.figure(figsize=(10,6))
sns.regplot(data = df, y = "High", x ="Volume", color = "b").set(title = "High vs Volume")


# In[ ]:


pip install scikit-learn


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df=df.iloc[1:]
df.tail()


# In[ ]:


train = df[0:1200]
valid = df[1200:]

x_train = train.iloc[:,[1,2,3]]
y_train = train.iloc[:,4]
date_train=train.iloc[:,0]
x_valid = valid.iloc[:,[1,2,3]]
y_valid = valid.iloc[:,4]
date_valid=valid.iloc[:,0]
date_train[1200]


# In[ ]:


plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(figsize = (8,3))
ax.plot(date_train, y_train, color = 'red', marker='', linewidth='0.75')
ax.plot(date_valid, y_valid, color = 'blue', marker='', linewidth='0.75')
plt.setp(ax.get_xticklabels(), rotation=45)
plt.legend(['Training Data', 'Testing Data'], loc='upper left')
ax.set(xlabel="Date",
       ylabel="Close Price",
       title="Stock Prices");
plt.show()


# In[ ]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)
pred = reg.predict(x_valid)


# In[ ]:


import math
err=[]
a=0
SUM=0
for i in range(1201,1693):
    a=y_valid[i]-pred[i-1201]
    err.append(a)
    SUM = SUM + pow(a,2)
(SUM/492)


# In[ ]:





# In[ ]:




