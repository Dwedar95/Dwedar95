# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 16:11:36 2022

@author: Moham
"""

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



import nltk 
nltk.download('stopwords')


##LOADING DATA to DATA FRAME PANDAS

news_data = pd.read_csv('train.csv')

news_data.head()  ##checking the first 5 objects

##replacing the NAN values with empty string beccause we have a lot of data and dont need to get the average

news_data = news_data.fillna('')


## Merging the author column and news titles in one column 

news_data['content'] = news_data['author']+''+news_data['title']

##DATA ANALYZING Sperating DATA and LABEL
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']

##Stemming 
##Stemming is to reduce the word to its root word

port_stem = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()  ##to make the alphapet lower case
    stemmed_content = stemmed_content.split() ##splitting the words
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')] ##
    stemmed_content = ' '.join(stemmed_content) ##merged the words with a space 
    return stemmed_content


news_data['content']= news_data['content'].apply(stemming)


#seperating data and labels 

X = news_data['content'].values
Y = news_data['label'].values

##converting the textual data to numerical data using VECTROZIER

vectroizer = TfidfVectorizer()
vectroizer.fit(X)

X = vectroizer.transform(X)  ##creating the shape of X as its alogrithim in order to make the prediction

##Analyzing data

X_train,Y_train,X_test,Y_test = train_test_split(X,Y, test_size=0.2,stratify=Y,random_state=2)

##Training the MODEL

mdl = LogisticRegression()
mdl.fit(X_train,Y_train)


##evaluation

##training data
X_train_predict = mdl.predict(X_train)
training_dataaccuracy = accuracy_score(X_train_predict,Y_train)



##testing data


X_test_predict = mdl.predict(X_test)
test_dataaccuracy = accuracy_score(X_test_predict,Y_test)



##Making a Predictive System
X_new = X_test[3]

prediction = model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')
  
  