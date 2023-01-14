# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 12:41:03 2023

@author: Mohamed DWEDAR
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn  as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

##importing data

car_DS = pd.read_csv('car data.csv')

##Checking NaN

car_DS.isnull().sum()


##Checking the no. of distribution of categorical data

car_DS.Fuel_Type.value_counts()

car_DS.Seller_Type.value_counts()

car_DS.Transmission.value_counts()

##Encoding The Data to Numeric

car_DS.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
car_DS.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)
car_DS.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)



##Splitting Data and label

X = car_DS.drop(['Car_Name','Selling_Price'],axis=1)   ##I don't need car names
Y = car_DS['Selling_Price']


##Training and Test data

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)

##Mdl Training Linear Regression

mdl = LinearRegression()

mdl.fit(X_train,Y_train)


##Evalute model (Score)-Prediction

training_data_predection = mdl.predict(X_train)

error_score = metrics.r2_score(Y_train, training_data_predection)

##Visualizing the Data

plt.scatter(Y_train,training_data_predection)


##Evaluate Test Data
Test_data_predection = mdl.predict(X_test)

error_score = metrics.r2_score(Y_test, Test_data_predection)
plt.scatter(Y_test,Test_data_predection)


##Trying Lasso Regression Model

mdlLS = Lasso()

mdlLS.fit(X_train,Y_train)


##Evalute model (Score)-Prediction

training_data_predection = mdlLS.predict(X_train)

error_score2 = metrics.r2_score(Y_train, training_data_predection)

##Visualizing the Data

plt.scatter(Y_train,training_data_predection)


##Evaluate Test Data
Test_data_predection = mdlLS.predict(X_test)

error_score3 = metrics.r2_score(Y_test, Test_data_predection)
plt.scatter(Y_test,Test_data_predection)

##Predective System

input_data = (2014,5.59,27000,0,0,0,0)

##changing the data into numpy 
input_data_als_numpy_array = np.asarray(input_data)

##reshape the data

input_data_reshape = input_data_als_numpy_array.reshape(1,-1)

predection = mdlLS.predict(input_data_reshape)

print(predection)
