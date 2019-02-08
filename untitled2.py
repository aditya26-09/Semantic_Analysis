# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 19:24:01 2019

@author: Aditya
"""
#creating datasets
import pandas as pd
dataset = pd.read_csv("train.tsv",delimiter="\t",quoting=3)
dataset = dataset.iloc[:,[2,3]]



import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus= []
for i in range(0,156060):
    review = re.sub('[^a-zA-Z]',' ',dataset['Phrase'][i])
    review = review.lower()
    
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = " ".join(review)
    corpus.append(review) 
    
    
#creating bag of words model    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=10000)
X = cv.fit_transform(corpus).toarray()
y =dataset.iloc[:,1].values

X=X.reshape(-1,156060,10000)
y = y.reshape((-1,1))
#classifying data
from keras.models import Sequential
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

classifier.add(Convolution1D(32,2,activation = 'relu'))
classifier.add(MaxPooling1D(pool_size=2))
classifier.add(Convolution1D(32,2,activation = 'relu'))
classifier.add(MaxPooling1D(pool_size=2))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 128,activation = 'relu'))
classifier.add(Dense(output_dim = 4,activation = 'softmax'))

classifier.compile(optimizer='adam',loss = 'categorical_crossentropy' ,metrics=['accuracy'])

classifier.fit(X,y,batch_size=10,nb_epoch=25)
