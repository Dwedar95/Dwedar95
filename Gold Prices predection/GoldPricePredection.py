# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 17:34:51 2023

@author: Mohamed Dwedar
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from sklearn import metrics


## reading data

Gold_DS = pd.read_csv('gld_price_data.csv')

Gold_DS.isnull().sum()
Gold_DS.shape


## Correlation 
#Positve-Negative (Via HEATMAP)

correlation = Gold_DS.corr()

plt.figure(figsize=(8,8))
sns.heatmap(correlation, cbar = True, square=True, fmt = '.1f',annot = True, annot_kws={'size':8})

# checking the distribution of the GLD Price
sns.distplot(Gold_DS['GLD'],color='green')

##Splitting the Data and Label
X = Gold_DS.drop(['Date','GLD'],axis=1)
Y = Gold_DS['GLD']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)

##*Model* Training:
##Random Forest Regressor
regressor = RandomForestRegressor(n_estimators=100)
# training the model
regressor.fit(X_train,Y_train)

# prediction on Test Data
test_data_prediction = regressor.predict(X_test)
# R squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", error_score)

Y_test = list(Y_test)

plt.plot(Y_test, color='blue', label = 'Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()