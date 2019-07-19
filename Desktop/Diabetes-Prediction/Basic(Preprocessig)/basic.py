# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:47:14 2019

@author: Lenovo
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
diabetes = pd.read_csv('diabetes.csv')
print(diabetes.columns)
diabetes.head()
print("dimension of diabetes data: {}".format(diabetes.shape))
print(diabetes.groupby('Outcome').size())
sns.countplot(diabetes['Outcome'],label="Count")
columns = list(diabetes)[0:-1] # Excluding Outcome column which has only 
diabetes[columns].hist(stacked=False, bins=100, figsize=(12,30), layout=(14,2)); 
# Histogram of first 8 columns
def plot_corr(df, size=11):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)

plot_corr(diabetes)

diabetes.info()