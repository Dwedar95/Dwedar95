# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 12:08:33 2023

@author: Mohamed DWEDAR
"""

import numpy as np
import pandas as pd
import seaborn as sns  ##for data visulization
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

####Data_Preprocessing####

Loan_data = pd.read_csv('Loans.csv') ##(Shape = 614,13)
##No of the missong values
Loan_data.isnull().sum()

##Drop all the missing values

Loan_data = Loan_data.dropna()

##Label ENCODING 'YES 1,NO0' The last coloumn is Yes, No so we need to replace

Loan_data.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)


Loan_data['Dependents'].value_counts()  ##to count the values on Dependant column

##in Dependant column we have +3 so we need to replace this data to another value to 4 as an example

Loan_data = Loan_data.replace(to_replace='3+',value = 4)


##Data Visulaization

#1_ Education_LOAN STATUS

sns.countplot(x='Education',hue='Loan_Status',data=Loan_data)

#2_Marriage_LOAN STATUS

sns.countplot(x='Married',hue='Loan_Status',data=Loan_data)

##Changing the (Converting TEXTING DATA to NUMERICAL VALUES)

Loan_data.replace({'Married':{'No':0,'Yes':1}},inplace=True)

Loan_data.replace({'Gender':{'Male':1,'Female':0}},inplace=True)

Loan_data.replace({'Self_Employed':{'No':0,'Yes':1}},inplace=True)

Loan_data.replace({'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2}},inplace=True)

Loan_data.replace({'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)

##Droping ID Column

Loan_data = Loan_data.drop(columns='Loan_ID',axis=1)


##Data Processing (Data and Label)

X = Loan_data.drop(columns='Loan_Status',axis=1)

Y = Loan_data['Loan_Status']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2,stratify=Y)


##Training the model SVM
#Classification
classifier = svm.SVC(kernel='linear')
#Fitting training
classifier.fit(X_train,Y_train)

##accuracy score evaluation

X_train_predection = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_predection, Y_train) ##comparing the predictied X to Original Label (Y_train)

X_test_predection = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_predection, Y_test)

##Predective System

input_data = (4,1,0,0,1,0,6000,0.0,141.0,360.0,1.0)
##changing the input data to numpy array
in_data_als_NParrat = np.asarray(input_data)

##reshaping the np array for predicting the one distance
in_data_reshaped = in_data_als_NParrat.reshape(1,-1)
prediction1=classifier.predict(in_data_reshaped)

if (prediction1[0] == 'No'):
    print ('No loan is givin')
else:
        print ('The Loan can be given')