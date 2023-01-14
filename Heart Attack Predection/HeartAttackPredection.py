# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:09:43 2023

@author: Mohamed Dwedar
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


##Loading Data

HA_Data = pd.read_csv('heart_disease_data.csv')

HA_Data.isnull().sum()


HA_Data.describe()

##Checking the distribution of the TARGET COLUMN  (Label)

HA_Data['target'].value_counts()
#1 is HeartAttacked
#0 is HealthyHEART

##Seperating Data and Labeling

X = HA_Data.drop(columns='target',axis=1)
Y = HA_Data['target']



##Training_Testing the DATA

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=1,stratify=Y)

##Training the ML Model

model = LogisticRegression()

model.fit(X_train,Y_train)

##Score of the model (Evaluation)==Predection Score Accuracy

#Accuracy of Training Data X_train

X_train_predection = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_predection, Y_train)


##Accuracy of Test Data

X_test_predection = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_predection, Y_test)



##Building an Predection System

Input_Data=(56,1,1,120,236,0,1,178,0,0.8,2,0,2)

##Change the input data to numpy array

input_data_als_numpyarr = np.asarray(Input_Data)

##Reshaping the numpy array for only one data shape not a dataset

input_data_reshaped =input_data_als_numpyarr.reshape(1,-1)
predection =model.predict(input_data_reshaped)

if (predection[0]==0):
    print('Heart is Healthy Enjoy Life')
else:
    print('The person may have a hear attack')
