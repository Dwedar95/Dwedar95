# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 11:33:26 2022

@author: Moham
"""

import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

so_data = pd.read_csv('sonar data.csv', header=None) 
so_data.head()  ##checking the heads of the data
so_data.shape
so_data.describe() ##gives you the describtion of the data
so_data.value_counts() ##count the variables of a specific location
so_data[60].value_counts() ##counting the data of the columns 60 (rock and mines values)
so_data.groupby(60).mean()   ##getting the mean valuse after grouping the column 60 (R,M,"Rock and Mine")


##Seperating the normal DATA X and the "Label data (R&M)" Y

X = so_data.drop(columns=60,axis=1)  ##DATA "deleting the last coloums
Y = so_data[60]

##Training Data
X_train, Y_train, X_test, Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)


##Model Training  (with training data and training label)

mdl = LogisticRegression()
mdl.fit(X_train,Y_train)


##Model Evaluation "Accuracy on training data"

X_train_prediction = mdl.predict(X_train)
training_dataX_accuracy = accuracy_score(X_train_prediction,Y_train)


##Model Evaluation "Accuracy on test data"

X_test_prediction = mdl.predict(X_test)
test_dataX_accuracy = accuracy_score(X_test_prediction,Y_test)

##Making a predictive model
input_data = (0.0223	0.0375	0.0484	0.0475	0.0647	0.0591	0.0753	0.0098	0.0684	0.1487	0.1156	0.1654	0.3833	0.3598	0.1713	0.1136	0.0349	0.3796	0.7401	0.9925	0.9802	0.889	0.6712	0.4286	0.3374	0.7366	0.9611	0.7353	0.4856	0.1594	0.3007	0.4096	0.317	0.3305	0.3408	0.2186	0.2463	0.2726	0.168	0.2792	0.2558	0.174	0.2121	0.1099	0.0985	0.1271	0.1459	0.1164	0.0777	0.0439	0.0061	0.0145	0.0128	0.0145	0.0058	0.0049	0.0065	0.0093	0.0059	0.0022)
##changing the input data to numpy array
in_data_als_NParrat = np.asarray(input_data)

##reshaping the np array for predicting the one distance
in_data_reshaped = in_data_als_NParrat.reshape(1,-1)
prediction1=mdl.predict(in_data_reshaped)

if (prediction1[0] == 'R'):
    print ('the object is Rock')
    else:
        print ('the object is Mine')
        
        