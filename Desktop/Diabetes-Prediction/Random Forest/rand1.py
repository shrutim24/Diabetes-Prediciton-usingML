# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 12:00:03 2019

@author: Lenovo
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
diabetes = pd.read_csv('diabetes.csv')
from sklearn.tree import DecisionTreeClassifier

features_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
predicted_class = ['Outcome']

X = diabetes[features_cols].values      # Predictor feature columns (8 X m)
Y = diabetes[predicted_class]. values   # Predicted class (1=True, 0=False) (1 X m)
split_test_size = 0.30

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=split_test_size, random_state=52)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(rf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(rf.score(X_test, y_test)))