# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 06:33:14 2024

@author: osama
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

d = pd.read_csv("data.csv")
print(d.describe)
print(d.isnull().sum())
d.drop(columns=["Unnamed: 32"], inplace=True)
print(d.isnull().sum())
x= d.iloc[:,:-1]
y = d.iloc[:,-1]
x.dtypes

x.shape
x_obj = x.select_dtypes(include=["object"])
x_nobj = x.select_dtypes(exclude=["object"])
x_obj.shape
le= LabelEncoder()
x_obj = le.fit_transform(x_obj)
x_obj=pd.DataFrame(x_obj)
x=pd.concat([x_nobj,x_obj],axis=1)

xtr,xts,ytr,yts=train_test_split(x,y,train_size=0.7)
xtr.dtypes
m = MLPClassifier(hidden_layer_sizes=(100,150,200), learning_rate="constant", learning_rate_init=0.0001, max_iter=5000)
xts.columns = xts.columns.astype(str)
xtr.columns = xtr.columns.astype(str)
ytr = ytr.astype(str)
yts=yts.astype(str)
m.fit(xtr,ytr)
