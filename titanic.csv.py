#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
     


# In[4]:


df=pd.read_csv("titanic.csv")
df.head()


# In[5]:


df.shape


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)


# In[10]:


df.head()


# In[11]:


data = df.drop('Survived',axis='columns')
target = df.Survived


# In[12]:


data.Sex=data.Sex.map({'male':1,'female':2})
data.Sex


# In[13]:


data.Age[:10]


# In[14]:


data.Age = data.Age.fillna(data.Age.mean())
data.Age


# In[15]:


data.head()


# In[34]:


plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='Age', bins=30, kde=True)
plt.title('Distribution of Passenger Ages')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[35]:


plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='Pclass', y='Survived', palette='Set1')
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()


# In[36]:


plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Sex', palette='Set2')
plt.title('Gender Distribution of Passengers')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


# In[37]:


plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='Sex', y='Survived', palette='pastel')
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.show()


# In[38]:


numerical_columns = ['Age', 'Fare']
sns.pairplot(df[numerical_columns])
plt.show()


# In[23]:


inputs = df.drop('Survived',axis="columns")
target = df["Survived"]
inputs.head()


# In[27]:


target.head()


# In[28]:


from sklearn.preprocessing import LabelEncoder
le_sex = LabelEncoder()
inputs["sex_n"] = le_sex.fit_transform(inputs["Sex"])
inputs_n = inputs.drop("Sex", axis="columns")
inputs_n.Age = inputs_n.Age.fillna(inputs_n.Age.mean())
inputs_n.head()


# In[29]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(inputs_n,target,test_size=0.2)
from sklearn import tree
model = tree.DecisionTreeClassifier()


# In[30]:


model.fit(x_train,y_train)


# In[31]:


model.score(x_test,y_test)


# In[32]:


from sklearn import tree
model = tree.DecisionTreeClassifier()     
model.fit(data, target)
model.score(data, target)


# In[ ]:




