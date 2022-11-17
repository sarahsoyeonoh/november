#IMPORT PACKAGES
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import lightgbm as lgb
from multiprocessing.sharedctypes import Value
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from multiprocessing.sharedctypes import Value
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error
from catboost import CatBoostClassifier, Pool

import xgboost
import shap

# import data 
data = pd.read_csv('/Users/sarahoh/Desktop/py_scripts/hello/220901/221102_dataset.csv')
X = data[['ethnicity', 'race','maternal_age','complications', 'type', 'prenatal_care', 'throughout', 'first_and_second_only', 'other_patterns', 'first_only']]
y = data['fas_f']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# create categorical variables 
X = pd.get_dummies(X, columns=['race', 'type', 'ethnicity'], drop_first=True)

#split the dataset into training (70%) and testing (30%) sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
classes = ['no FAS', 'FAS']

# 3) XGBoost 
# create model instance
bst = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, objective='binary:logistic')
# fit model
bst.fit(X_train, y_train)
# make predictions
y_pred = bst.predict(X_test)

# compute SHAP values
explainer = shap.Explainer(bst, X)
shap_values = explainer(X)
shap.plots.bar(shap_values)
shap.plots.bar(shap_values, max_display=40)

