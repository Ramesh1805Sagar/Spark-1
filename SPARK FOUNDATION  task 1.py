#!/usr/bin/env python
# coding: utf-8

# # Task-1 Data Science and Bussiness Analytics Intership
# 
# 
# 

# # By Ramesh Ravindra Sagar Intern at Spark Foundation

# In[44]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.formula.api as snf
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[45]:


url="http://bit.ly/w-data"
df=pd.read_csv(url)


# In[46]:


df


# In[47]:


df.head()
df.tail()


# # Explorotery Data Analysis

# In[48]:


df.columns


# In[49]:


df.dtypes


# In[50]:


df.info()


# In[51]:


df.describe()


# In[52]:


df.corr()


# # Remove The Outliers

# In[53]:


def detection_outlier(df):
    num_columns=[]
    
    count=0
    y=[]
    for i in num_columns:
        z=np.abs(stats.zscore(df[i]))
        for j in range(len(z)):
            if z[j]>3 or z[j]<-3:
                t.append(j)
                count+=1
    df=df.drop(list(set(y)))
    df=df.reset_index()
    df=df.drop('index',axis=1)
    print(count)
    return df


# In[54]:


df=detection_outlier(df)


# # Distriution

# In[55]:


sns.distplot(df["Scores"])
plt.show()

sns.distplot(df["Scores"],kde=False,rug=True)
plt.show()



# In[56]:


sns.jointplot(df["Hours"],df["Scores"],kind="reg").annotate(stats.pearsonr)
plt.show()


# # Performing simple Linear Regression

# # Calculating the coefficient of the simple loinear regression eqaution :y=Bo+B1.x(B1:slope,Bo:intercept)

# In[57]:


mean_x = np.mean(df["Hours"])
mean_y = np.mean(df["Scores"])
num = 0
den = 0
x = list(df["Hours"])
y = list(df["Scores"])
for i in range(len(df)):
    num += (x[i]-mean_x)*(y[i]-mean_y)
    den += (x[i]-mean_x)**2
B1 = num/den
B1


# In[58]:


Bo=mean_y-B1*mean_x
Bo


# # Making a prediction

# In[59]:


df["prediction_scores"]=Bo + B1*df["Hours"]
df.head()


# In[60]:


plt.scatter(df["Hours"],df["Scores"])
plt.scatter(df["Hours"],df["prediction_scores"])
plt.plot()


# # prediction of an given value 9.25

# In[61]:


Bo+B1*9.25


# In[62]:


y=list(df["Scores"].values)
y_pred=list(df["prediction_scores"].values)


# # RMSE

# In[63]:


s=sum([(y_pred[i]-y[i])**2 for i in range(len(df))])
rmse=(np.sqrt(s/len(df)))/mean_y
rmse


# # OLS Model

# In[64]:


model = snf.ols('Scores~Hours',data=df)
model=model.fit()


# In[65]:


df['pred_ols']=model.predict(df['Hours'])
plt.figure(figsize=(12,6))
plt.plot(df['Hours'],df['pred_ols'])               #regression line
plt.plot(df['Hours'],df['Scores'],'ro')          #scatter plot showing actual data
plt.title('Actual vs Predicted')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


# # In this scatterplot we can easy to observed the predicted value for 9.25 hourse is around 92

# # Conclusion in this case we can say categorical prediction

# In[66]:


# consider a threshold to come a conclusion whether the student passed or fail
# we can consider here 38 as the cutoff to pass.
cut_off=38
df["Passed?"]=df["Scores"]>=40
df.head()


# # plotting the datas results

# In[67]:


sns.countplot(df["Passed?"])


# # Feature Engineering

# In[68]:


feature=df["Hours"].values.reshape(-1,1)
target=df["Scores"].values


# # Splitting the data

# In[69]:


x_train,x_test,y_train,y_test=train_test_split(feature,target,random_state=0)


# # Training the regression model

# In[70]:


from sklearn import linear_model 
ln=linear_model.LinearRegression()
ln.fit(x_train,y_train)


# # Accuracy

# In[71]:


ln.score(x_train,y_train)


# In[72]:


ln.score(x_test,y_test)


# # Predicting the outcomes

# In[73]:


results=[[9.25]]
ln.predict(results)


# In[ ]:




