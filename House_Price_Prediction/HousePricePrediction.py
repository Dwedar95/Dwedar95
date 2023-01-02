# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 11:45:26 2022

@author: Moham
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets 
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


House_P_DS = sklearn.datasets.load_boston()
print(House_P_DS)

##Loading the Dataaset to pandas DATAFRAME
##While the target Price column is not included so we have to include it
House_P_DS_DF = pd.DataFrame(House_P_DS.data,columns=House_P_DS.feature_names) ##here the data is seperated into data feature names and target


House_P_DS_DF['Price']=House_P_DS.target

##UNDERSTANDING THE COORELATION BETWEEN VARIOUS FEATURES

#Positive Corellation
#Negative Corellation
##To understand it we need to set up the heatmap
correlation = House_P_DS_DF.corr()   


plt.figure(figsize=(10,10))
sns.heatmap(correlation, cmap='Blues',cbar=True,fmt = '.1f',annot_kws={'size':8})   


##Splitting the Data and Label "Prices"


X = House_P_DS_DF.drop(['Price'],axis =1)
Y = House_P_DS_DF['Price']


##DATA ANALSYING

X_train,X_test,Y_train,Y_test = train_test_split(X,Y , test_size=0.2, random_state=2)


##Model Fitting XGBOOST REGRESSION

mdl = XGBRegressor()  ##loading the model
mdl.fit(X_train,Y_train)  ##training the model

##Prediction the data (training data)

training_data_predection = mdl.predict(X_train)


##ACCURACY LOSS calculating

score1=metrics.r2_score(Y_train,training_data_predection)  ##R SQUARED METHOD

score2=metrics.mean_absolute_error(Y_train,training_data_predection)  ##Mean values


##visualizing the data

plt.scatter(Y_train, training_data_predection)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.show()













