## Packages
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from collections import Counter


## Load data
X_train = pd.read_parquet("C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/processed/X_train_DzY_DummyN_SsN.parquet")
y_train = pd.read_csv("C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/processed/y_train_DzY_DummyN_SsN.csv")
y_train = np.ravel(y_train, order='C')

## Model training
from interpret.glassbox import ExplainableBoostingClassifier
model_ebm = ExplainableBoostingClassifier(random_state=42)  
scoring = ["precision", "average_precision", "recall"]
cv_result = cross_validate(model_ebm, X_train, y_train, scoring=scoring, n_jobs=-2, verbose=2)


## Hyperparameter tuning Grid search
params = {
    'learning_rate': [0.01, 0.1, 0.5],
    'max_bins': [128, 256, 512],
    'interactions': [5, 10, 15],
    'max_rounds': [2000, 5000, 10000],
    'min_samples_leaf': [1, 2, 5],
    'max_leaves': [2, 4, 8],
}

CV_ebm = GridSearchCV(model_ebm, param_grid=params, cv = 5, scoring='precision', verbose=2, refit=True, n_jobs=-2)
CV_ebm.fit(X_train, y_train) # RUN DEZE 1 KEER!!!!


model_ebm2 = ExplainableBoostingClassifier(random_state=42, learning_rate=0.01, max_bins=200, interactions=10, max_rounds=5000, min_samples_leaf=2, max_leaves=3, n_jobs=-2)   
model_ebm2.fit(X_train, y_train)

# save the model to disk
import pickle
filename = 'C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/models/model_ebm_DzY.sav'
pickle.dump(model_ebm2, open(filename, 'wb'))


