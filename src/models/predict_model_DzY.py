## Packages
import pandas as pd
import numpy as np
import pickle
import interpret.glassbox
from sklearn.metrics import confusion_matrix
from sklearn import metrics

## Load model
filename = 'C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/models/model_ebm_DzY.sav'
model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)

## Load data
X_test = pd.read_parquet("C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/processed/X_test_DzY_DummyN_SsN.parquet")
y_test = pd.read_csv("C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/processed/y_test_DzY_DummyN_SsN.csv")
y_test = np.ravel(y_test, order='C')
X_vali = pd.read_parquet("C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/processed/X_vali_DzY_DummyN_SsN.parquet")
y_vali = pd.read_csv("C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/processed/y_vali_DzY_DummyN_SsN.csv")
y_vali = np.ravel(y_vali, order='C')


## Predict het model op de testset
preds = model.predict(X_test)
precision = metrics.precision_score(y_test, preds)
print(f"Precision =  {(precision * 100).round(1)}")
print(confusion_matrix(y_test, preds))

## Tunen met moving thresholds
pred_proba = model.predict_proba(X_test)
threshold = 0.6
preds = (pred_proba[:,1] >= threshold).astype('int')
precision = metrics.precision_score(y_test, preds)
print(f"Precision =  {(precision * 100).round(1)}")
print(confusion_matrix(y_test, preds)) #geen goede resultaten


## Als laatste een test op de validatie set
preds = model.predict(X_vali)
precision = metrics.precision_score(y_vali, preds)
print(f"Precision =  {(precision * 100).round(1)}")
print(confusion_matrix(y_vali, preds))







