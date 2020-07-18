# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 17:55:50 2020

@author: Dell
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
ds=pd.read_csv("employee.csv")
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
c=[]
for i in range(0,210):
    review = re.sub('[^a-zA-Z]', ' ', ds['feedback'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    c.append(review)
from sklearn.feature_extraction.text import CountVectorizer
cv =CountVectorizer(max_features=100)
x=cv.fit_transform(c).toarray()
y=ds.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
y_train.shape
import pickle
pickle.dump(cv.vocabulary_,open("feature.pkl","wb"))
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model=Sequential()
model.add(Dense(input_dim=34,kernel_initializer="random_uniform",activation="sigmoid",units=1000))
model.add(Dense(kernel_initializer="random_uniform",activation="sigmoid",units=100))
model.add(Dense(units=1,kernel_initializer='random_uniform',activation="sigmoid"))
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
model.summary()
model.fit(x_train,y_train,epochs=50,batch_size=1)

y_pred=model.predict(x_test)
y_pred=(y_pred>0.5)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
loaded=CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl","rb")))
da="bad quality"
da=da.split("delimeter")
result=model.predict(loaded.transform(da))
print(result)
prediction=result>0.5
print(prediction)


