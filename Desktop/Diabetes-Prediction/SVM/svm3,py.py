# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 12:10:27 2019

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
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
from sklearn.svm import SVC
svc = SVC(C=1000)
svc.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(
    svc.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))

y_pred=svc.predict(X_test_scaled)
y_pred_list=list(y_pred)
y_train_test=list(y_test)

tp=0
tn=0
fp=0
fn=0

for x in range(0,len(y_pred)):
    if y_pred_list[x]==1 and y_train_test[x]==1:
        tp=tp+1
    if y_pred_list[x]==0 and y_train_test[x]==0:
        tn=tn+1
    if y_pred_list[x]==1 and y_train_test[x]==0:
        fp=fp+1
    if y_pred_list[x]==0 and y_train_test[x]==1:
        fn=fn+1
        
print('Confusio Matrix:')
print('\t\tActual values')
print('pred\t1\t',tp,'\t',fp)
print('iceted\t0\t',fn,'\t',tn)

acc=(tp+tn)/(tp+fp+tn+fn)
print('\nAccuracy: ',acc)

print('Misclassification: ',1-acc)
print('Senstitivity: ',tp/(tp+fn))
print('Specificity: ',tn/(fp+tn))
prec=tp/(tp+fp)
recall=tp/(tp+fn)
print('Precision: ',prec)
print('Recally: ',recall)
print('Fscore: ',2*prec*recall/(prec+recall))