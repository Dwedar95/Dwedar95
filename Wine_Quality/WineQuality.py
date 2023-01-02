# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 12:54:13 2023

@author: Moham
"""

##Importing Libariries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns



## Loading Data

Wine_DS = pd.read_csv('RedWineQuality.csv')

##Checking the NaN

Wine_DS.isnull().sum()

##Data Analsyis

##plotting the values of each quality

sns.catplot(x='quality',data=Wine_DS,kind='count')

##plotting the volatile acidity column to the quality column

plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='volatile acidity',data = Wine_DS)

##plotting the citric acidity column to the quality column

plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='citric acid',data = Wine_DS)


## AND SO ON .....



##CORRELATION defining the HEATMAP to understand the correlation between columns


#Positive Corelation or Negative Correlation
correlation = Wine_DS.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')


##Data preprocessing seperating data and label

X = Wine_DS.drop('quality',axis=1)
Y = Wine_DS['quality']

##Changing the quality label values from 0 - 7, to 1,0
##Label Binary Session
Y = Wine_DS['quality'].apply(lambda y_value:1 if y_value>=7 else 0)

##Training and splitting DATA

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2,stratify=Y)


##Model Training
#Random Forest Classifier

mdl = RandomForestClassifier()

mdl.fit(X_train, Y_train)

##Model Evaluation

#Accuracy on Test Data

X_test_predection = mdl.predict(X_test)

test_data_accuracy = accuracy_score(X_test_predection, Y_test)




##Predective System

input_data = (7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4)

##changing the data into numpy 
input_data_als_numpy_array = np.asarray(input_data)

##reshape the data

input_data_reshape = input_data_als_numpy_array.reshape(1,-1)

predection = mdl.predict(input_data_reshape)

print(predection)

if (predection[0]==1):
    print('The wine is good')
else:
    print('Bad Quality Wine')