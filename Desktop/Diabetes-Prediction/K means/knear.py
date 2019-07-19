# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:53:08 2019

@author: Lenovo
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
diabetes = pd.read_csv('diabetes.csv')
training_accuracy = []
test_accuracy = []
features_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
predicted_class = ['Outcome']

X = diabetes[features_cols].values      # Predictor feature columns (8 X m)
Y = diabetes[predicted_class]. values   # Predicted class (1=True, 0=False) (1 X m)
split_test_size = 0.30

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=split_test_size, random_state=52)
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))