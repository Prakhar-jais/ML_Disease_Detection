# -*- coding: utf-8 -*-
"""HeartDiseaseDetection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1W8JNQf8ZSPxNuxjuoK2BSqWl5yLTR1EN
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('/content/heart_disease_data.csv')

data.head()

data.info()

data.describe()

data.isnull().sum()

# check for distribution of Target Variable
data['target'].value_counts()

# 1 --> Defective Heart

# 0 --> Healthy Heart

DWT = data.drop(columns='target', axis=1)
T = data['target']

print(DWT)

print(T)

X_train, X_test, Y_train, Y_test = train_test_split(DWT, T, test_size=0.2, stratify=T, random_state=2)

print(DWT.shape, X_train.shape, X_test.shape)

model = LogisticRegression()

model.fit(X_train, Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data : ', test_data_accuracy)

input_data = (56,1,1,120,236,0,1,178,0,0.8,2,0,2)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')

#  save my model 
import pickle

filename = 'Heart_disease_Model.sav'
pickle.dump(model,open(filename,'wb'))

# load model
load_m = pickle.load(open('Heart_disease_Model.sav','rb'))

