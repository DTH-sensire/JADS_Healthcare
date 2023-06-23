## packages
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from collections import Counter

#Load data:
X_train = pd.read_parquet("C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/processed/X_train_DzY_DummyN_SsN.parquet")
y_train = pd.read_csv("C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/processed/y_train_DzY_DummyN_SsN.csv")
y_train = np.ravel(y_train, order='C')
X_test = pd.read_parquet("C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/processed/X_test_DzY_DummyN_SsN.parquet")
y_test = pd.read_csv("C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/processed/y_test_DzY_DummyN_SsN.csv")
y_test = np.ravel(y_test, order='C')
X_vali = pd.read_parquet("C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/processed/X_vali_DzY_DummyN_SsN.parquet")
y_vali = pd.read_csv("C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/processed/y_vali_DzY_DummyN_SsN.csv")
y_vali = np.ravel(y_vali, order='C')

## Tuning voor Support Vector Machine Linear
from sklearn.svm import LinearSVC
model_lsvc = LinearSVC(random_state=42, dual = False, max_iter=2000)
scoring = ["precision", "average_precision", "recall"]
cv_result = cross_validate(model_lsvc, X_train, y_train, scoring=scoring, verbose=2, n_jobs=-2)
cv_result

model_lsvc.fit(X_train, y_train)
preds = model_lsvc.predict(X_test)
precision = metrics.precision_score(y_test, preds)
print(f"Precision =  {(precision * 100).round(1)}")
print(confusion_matrix(y_test, preds))

## SVM met undersampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_train, y_train)
cv_result = cross_validate(model_lsvc, X_res, y_res, scoring=scoring)
cv_result

model_lsvc.fit(X_res, y_res)
preds = model_lsvc.predict(X_test)
precision = metrics.precision_score(y_test, preds)
print(f"Precision =  {(precision * 100).round(1)}")
print(confusion_matrix(y_test, preds))

## Moving thresholds
from sklearn.calibration import CalibratedClassifierCV
model_lsvc2 = CalibratedClassifierCV(model_lsvc) #dit moet omdat lsvc niet kan omgaan met predcit_proba
model_lsvc2.fit(X_res, y_res)
pred_proba = model_lsvc2.predict_proba(X_test)
threshold = 0.8
preds = (pred_proba[:,1] >= threshold).astype('int')
precision = metrics.precision_score(y_test, preds)
print(f"Precision =  {(precision * 100).round(1)}")
print(confusion_matrix(y_test, preds))

## Nu een RF met finetuning, aangezien svm fine tuning niet kan
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(random_state=42, class_weight='balanced_subsample', max_features = 'log2', n_jobs= -2, criterion='gini', max_depth=None, max_leaf_nodes=None)
model_rf.fit(X_train, y_train)

params = { 
    'n_estimators' :[150, 200, 250],
    'max_features' :['log2']
        } # misschien nog n_estimators grid rond de 200

CV_brf = GridSearchCV(model_rf, param_grid=params, cv = 5, scoring='f1', verbose=2, refit=True)
CV_brf.fit(X_train, y_train) # RUN DEZE 1 KEER!!!!

CV_brf.best_params_
CV_brf.best_estimator_ 

model_rf2 = RandomForestClassifier(random_state=42, class_weight='balanced_subsample', n_estimators=200, max_features = 'log2', criterion = 'gini', max_depth = None, max_leaf_nodes = None, n_jobs= -2)
cv_result = cross_validate(model_rf2, X_train, y_train, scoring=scoring, verbose=2, n_jobs=-2)
cv_result

model_rf2.fit(X_train, y_train)
preds = model_rf2.predict(X_test)
precision = metrics.precision_score(y_test, preds)
print(f"Precision =  {(precision * 100).round(1)}")
print(confusion_matrix(y_test, preds))

# DZ N
## Precision = 74.2
#[[8784   37]
# [1548  107]]

# DZ Y
# Precision =  63.5
# [[9245   61]
# [1655  106]]

## Moving thresholds
pred_proba = model_rf2.predict_proba(X_test)
threshold = 0.45
preds = (pred_proba[:,1] >= threshold).astype('int')
precision = metrics.precision_score(y_test, preds)
print(f"Precision =  {(precision * 100).round(1)}")
print(confusion_matrix(y_test, preds))

# DZ N
# Precision =  66.4 met threshold 0.45
# [[8749   72]
# [1513  142]]


## Feature selection op model_rf2
from sklearn.feature_selection import RFECV
selector = RFECV(model_rf2, step=1, cv=5)
selector = selector.fit(X_train, y_train, verbose=2, n_jobs= -2)


mask = selector.get_support()
features = X_train.columns 
best_features = features[mask]

print("All features: ", X_train.shape[1])
print(features)

print("Selected best: ", best_features.shape[0])
print(features[mask]) 

print("Verwijderde features: ", (X_train.shape[1] - best_features.shape[0]))
print(features[~np.array(mask)])

model_rf3 = RandomForestClassifier(random_state=42, class_weight='balanced_subsample', n_estimators=200, max_features = 'log2', criterion = 'gini', max_depth = None, max_leaf_nodes = None, n_jobs= -2)
model_rf3.fit(X_train[best_features], y_train)

cv_result = cross_validate(model_rf3, X_train[best_features], y_train, scoring=scoring)

preds = model_rf3.predict(X_test[best_features])
precision = metrics.precision_score(y_test, preds)
print(f"Precision =  {(precision * 100).round(1)}")
print(confusion_matrix(y_test, preds))



## Geslaagde/niet-geslaagde behandeluitkomst

from sklearn.metrics import get_scorer_names
get_scorer_names()


## Alleen nog een BRF met moving thresholds
from imblearn.ensemble import BalancedRandomForestClassifier
model_brf = BalancedRandomForestClassifier(random_state=42, class_weight='balanced_subsample', n_estimators=200, max_features = 'log2', criterion = 'gini', max_depth = None, max_leaf_nodes = None, n_jobs= -2)
cv_result = cross_validate(model_brf, X_res, y_res, scoring=scoring)
cv_result

model_brf.fit(X_res, y_res)
preds = model_brf.predict(X_test)
precision = metrics.precision_score(y_test, preds)
print(f"Precision =  {(precision * 100).round(1)}")
print(confusion_matrix(y_test, preds)) # NIET GOED

## Moving thresholds
pred_proba = model_brf.predict_proba(X_test)
threshold = 0.9
preds = (pred_proba[:,1] >= threshold).astype('int')
precision = metrics.precision_score(y_test, preds)
print(f"Precision =  {(precision * 100).round(1)}")
print(confusion_matrix(y_test, preds)) #NIET GOED




## Explainable boosting
from interpret.glassbox import ExplainableBoostingClassifier
model_ebm = ExplainableBoostingClassifier(random_state=42)  
cv_result = cross_validate(model_ebm, X_train, y_train, scoring=scoring)


ebm.fit(X_train, y_train)
