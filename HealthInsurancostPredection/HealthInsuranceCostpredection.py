# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 18:24:31 2023

@author: Moham
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import metrics


### Data loading

I_DS = pd.read_csv('insurance.csv')

##distripution of age values

sns.set()
plt.figure(figsize=(6,6))
sns.distplot(I_DS['age'])
plt.title('Age Distribution')
plt.show()


##Gender Column

plt.figure(figsize=(6,6))
sns.countplot(x='sex',data=I_DS)
plt.title('Sex Distribution')
plt.show()

##BMI distribution 
sns.set()
plt.figure(figsize=(6,6))
sns.distplot(I_DS['bmi'])
plt.title('Age Distribution')
plt.show()


##Childern plot *Numbers* so we need countplot*

plt.figure(figsize=(6,6))
sns.countplot(x='children',data=I_DS)
plt.title('Children Distribution')
plt.show()

##Smoker columns 
plt.figure(figsize=(6,6))
sns.countplot(x='smoker',data=I_DS)
plt.title('smoker numbers')
plt.show()


##region columns 
plt.figure(figsize=(6,6))
sns.countplot(x='region',data=I_DS)
plt.title('region numbers')
plt.show()


##charges distribution 
sns.set()
plt.figure(figsize=(6,6))
sns.distplot(I_DS['charges'])
plt.title('Charges Distribution')
plt.show()


###Data Preprocessing

#We Have 3 Categorical columns (Non numeric) (Sex,smoker and Region ) ##ENCODING

I_DS.replace({'sex':{'male':0,'female':1}},inplace=True)
I_DS.replace({'region':{'northwest':0,'northeast':1,'southeast':2,'southwest':4}},inplace=True)
I_DS.replace({'smoker':{'no':0,'yes':1}},inplace=True)


##Splitting the Label and Data

X = I_DS.drop(columns='charges',axis = 1)
Y = I_DS['charges']



##Training and testing Data

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)


##Linear Regression Model

mdl = LinearRegression()

mdl.fit(X_train,Y_train)

##Socring the model and predection

training_data_predection = mdl.predict(X_train)
Test_data_predection = mdl.predict(X_test)

##R square value

R2_Train = metrics.r2_score(Y_train,training_data_predection)

R2_Test = metrics.r2_score(Y_test, Test_data_predection)




##Building a predictive model

input_data = (19,1,27.9,0,1,4)

input_data_als_array = np.asarray(input_data)
##reshaping 
input_data_als_array_reshaped = input_data_als_array.reshape(1,-1)

##prediction

prediction = mdl.predict(input_data_als_array_reshaped)

 













    