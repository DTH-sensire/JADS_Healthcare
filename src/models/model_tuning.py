## packages
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

#Load data:
X_train = pd.read_parquet("C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/processed/X_train_DzN_DummyY_SsY.parquet")
y_train = pd.read_csv("C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/processed/y_train_DzN_DummyY_SsY.csv")
y_train = np.ravel(y_train, order='C')
X_test = pd.read_parquet("C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/processed/X_test_DzN_DummyY_SsY.parquet")
y_test = pd.read_csv("C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/processed/y_test_DzN_DummyY_SsY.csv")
y_test = np.ravel(y_test, order='C')
X_vali = pd.read_parquet("C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/processed/X_vali_DzN_DummyY_SsY.parquet")
y_vali = pd.read_csv("C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/processed/y_vali_DzN_DummyY_SsY.csv")
y_vali = np.ravel(y_vali, order='C')

## Tuning voor Support Vector Machine
from sklearn.svm import LinearSVC
model_lsvc = LinearSVC(random_state=42, dual = False, max_iter=2000)
model_lsvc.fit(X_train, y_train)
preds = model_lsvc.predict(X_test)
precision = metrics.precision_score(y_test, preds)
print(f"Precision =  {(precision * 100).round(1)}")
print(confusion_matrix(y_test, preds))

## Nu een SVC met meer params om te tunen, conclusie: SVC duurt te lang. Te complexe algo voor deze data. Met alleen linearSVC kan ik niet tunen, dus dan de RF verder tunen
from sklearn.svm import SVC
model_svc = SVC(kernel='linear', random_state=42)
model_svc.fit(X_train, y_train)

params = {
    'C': [0.1,1, 10, 100], 
    'gamma': [1,0.1,0.01,0.001],
    'kernel': ['linear','rbf', 'poly', 'sigmoid']
    }

CV_brf = GridSearchCV(model_svc, param_grid=params, cv = 5, scoring='precision', verbose=2, refit=True)
CV_brf.fit(X_train, y_train)
CV_brf.best_estimator_ 

