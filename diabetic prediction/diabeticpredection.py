# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 12:25:55 2022

@author: Moham
"""
##importing libraries
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

##importing the data
diabetes_dataset = pd.read_csv('diabetes.csv')

diabetes_dataset.shape
diabetes_dataset.describe()
diabetes_dataset['Outcome'].value_counts() ##the column outcome is the results colums
##counting the values of (0,"Non diabetic",1 "diabetic")

diabetes_dataset.groupby('Outcome').mean()  ##very important getting the mean of the results column

##Seperating DATA "X" and Label "Y"

X = diabetes_dataset.drop(columns = 'Outcome', axis = 1)
Y = diabetes_dataset['Outcome']

 
##Data Scalling and standriezing

scaler = StandardScaler()  ##Scaling the data
scaler.fit(X)  ##fiting the scaled data "X"

standarized_data = scaler.transform(X) ## we can use this function as well


X = standarized_data
Y = diabetes_dataset['Outcome']

##Data Training and Testing

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

##Training the model

classifier = svm.SVC(kernel ='linear')

classifier.fit(X_train,Y_train)


##Model Evalution

##Accuracy 

X_train_prediction = classifier.predict(X_train)

training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print(training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(test_data_accuracy)


##Making a predictive System

input_data = (6,148,72,35,0,33.6,0.627,50)

##change it into numpy

input_data_as_NUMPY = np.asarray(input_data)

##reshape the array
input_data_reshape = input_data_as_NUMPY.reshape(1,-1)

##Standrized the input data

std_data = scaler.transform(input_data_reshape)

print(std_data)

##prediction

prediction = classifier.predict(std_data)
print(prediction)
