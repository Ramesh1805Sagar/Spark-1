#!/usr/bin/env python
# coding: utf-8

# # Task 2- Data Science and Bussiness Analytics Intership

# # Unsupervised Learning (K Means Clustering) - Iris Dataset

# # K- Means Clustering

# # by By Sagar Ramesh Ravindra Intern at Spark Foundation

# # Importing the libraries

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


Iris =pd.read_csv(r'C:\Users\91830\Downloads\Iris.csv')


# In[4]:


Iris.info()


# In[5]:


Iris.describe()


# # Dividig this into Independent and dependent features

# In[6]:


x=Iris.iloc[:, [1,4]].values


# # Using the elbow method to find the optmimal number of clusters

# In[8]:


from sklearn.cluster import KMeans
wcss=[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init ='k-means++', random_state =42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# # Training the KMeans model on the dataset

# In[9]:


kmeans = KMeans(n_clusters = 3, init ='k-means++', random_state = 42)
y_kmeans=kmeans.fit_predict(x)


# In[10]:


print(y_kmeans)


# In[11]:


plt.scatter(x[y_kmeans ==0, 0], x[y_kmeans==0, 1], s=100, c='pink', label ='Iris-setosa')
plt.scatter(x[y_kmeans ==1, 0], x[y_kmeans==1, 1], s=100, c='yellow', label ='Iris-versicolour')
plt.scatter(x[y_kmeans ==2, 0], x[y_kmeans==2, 1], s=100, c='blue', label ='Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='pink', label ='Centroids')
plt.title('Cluster of Iris Data')
plt.xlabel('Sepal Length', fontsize =18)
plt.ylabel('Sepal Width', fontsize =18)
plt.legend()
plt.show()


# In[ ]:




