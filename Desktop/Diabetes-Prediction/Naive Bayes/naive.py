# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 12:26:25 2019

@author: Lenovo
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
diabetes = pd.read_csv('diabetes.csv')
features_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
predicted_class = ['Outcome']

X = diabetes[features_cols].values      # Predictor feature columns (8 X m)
Y = diabetes[predicted_class]. values   # Predicted class (1=True, 0=False) (1 X m)
split_test_size = 0.30

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=split_test_size, random_state=52)



diab_model = GaussianNB()

diab_model.fit(X_train, y_train.ravel())
diab_train_predict = diab_model.predict(X_train)

from sklearn import metrics


print("Accuracy on training set: {0:.3f}".format(metrics.accuracy_score(y_train, diab_train_predict)))



diab_test_predict = diab_model.predict(X_test)

from sklearn import metrics

print("Accuracy on test set: {0:.3f}".format(metrics.accuracy_score(y_test, diab_test_predict)))
