
## packages
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix

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

## Tabel maken voor alle preds
index = []
scores = {'Precision':[], "Average_Precision":[], "Recall":[]}
scoring = ["precision", "average_precision", "recall"]

## Dummy Classifier
from sklearn.dummy import DummyClassifier
model_naive = DummyClassifier(strategy='most_frequent', random_state=42) #random class
cv_result = cross_validate(model_naive, X_train, y_train, scoring=scoring)

index += ["Dummy Classifier"]
scores["Precision"].append(cv_result["test_precision"].mean())
scores["Average_Precision"].append(cv_result["test_average_precision"].mean())
scores["Recall"].append(cv_result["test_recall"].mean())

df_output = pd.DataFrame(scores, index=index)
df_output

## Logistic Regression
from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression(max_iter=1000, random_state=42)
cv_result = cross_validate(model_lr, X_train, y_train, scoring=scoring)

index += ["Logistic Regression"]
scores["Precision"].append(cv_result["test_precision"].mean())
scores["Average_Precision"].append(cv_result["test_average_precision"].mean())
scores["Recall"].append(cv_result["test_recall"].mean())

df_output = pd.DataFrame(scores, index=index)
df_output

## Random Forest
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(random_state=42)
cv_result = cross_validate(model_rf, X_train, y_train, scoring=scoring)

index += ["Random Forest"]
scores["Precision"].append(cv_result["test_precision"].mean())
scores["Average_Precision"].append(cv_result["test_average_precision"].mean())
scores["Recall"].append(cv_result["test_recall"].mean())

df_output = pd.DataFrame(scores, index=index)
df_output

## Random Forest met balanced_subsample
from sklearn.ensemble import RandomForestClassifier
model_rf_bs = RandomForestClassifier(random_state=42, class_weight='balanced_subsample')
cv_result = cross_validate(model_rf_bs, X_train, y_train, scoring=scoring)

index += ["Random Forest met Balanced Subsample"]
scores["Precision"].append(cv_result["test_precision"].mean())
scores["Average_Precision"].append(cv_result["test_average_precision"].mean())
scores["Recall"].append(cv_result["test_recall"].mean())

df_output = pd.DataFrame(scores, index=index)
df_output

## Random Forest met balanced class_weight
from sklearn.ensemble import RandomForestClassifier
model_rf_b = RandomForestClassifier(random_state=42, class_weight='balanced')
cv_result = cross_validate(model_rf_b, X_train, y_train, scoring=scoring)

index += ["Random Forest met Balanced"]
scores["Precision"].append(cv_result["test_precision"].mean())
scores["Average_Precision"].append(cv_result["test_average_precision"].mean())
scores["Recall"].append(cv_result["test_recall"].mean())

df_output = pd.DataFrame(scores, index=index)
df_output

## Beste Random Forest met random undersampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_train, y_train)
cv_result = cross_validate(model_rf_bs, X_res, y_res, scoring=scoring)

index += ["Random Forest Balanced Subsample + Under-sampling"]
scores["Precision"].append(cv_result["test_precision"].mean())
scores["Average_Precision"].append(cv_result["test_average_precision"].mean())
scores["Recall"].append(cv_result["test_recall"].mean())

df_output = pd.DataFrame(scores, index=index)
df_output

## Balanced Random Forest
from imblearn.ensemble import BalancedRandomForestClassifier
model_brf = BalancedRandomForestClassifier(random_state=42)
cv_result = cross_validate(model_brf, X_train, y_train, scoring=scoring)

index += ["Balanced Random Forest"]
scores["Precision"].append(cv_result["test_precision"].mean())
scores["Average_Precision"].append(cv_result["test_average_precision"].mean())
scores["Recall"].append(cv_result["test_recall"].mean())

df_output = pd.DataFrame(scores, index=index)
df_output

## Log Regression met under-sampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_train, y_train)
cv_result = cross_validate(model_lr, X_res, y_res, scoring=scoring)

index += ["Logistic Regression + under-sampling"]
scores["Precision"].append(cv_result["test_precision"].mean())
scores["Average_Precision"].append(cv_result["test_average_precision"].mean())
scores["Recall"].append(cv_result["test_recall"].mean())

df_output = pd.DataFrame(scores, index=index)
df_output

##  Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
model_gbc = GradientBoostingClassifier(random_state=42)
cv_result = cross_validate(model_gbc, X_train, y_train, scoring=scoring)

index += ["Gradient Boosting Classifier"]
scores["Precision"].append(cv_result["test_precision"].mean())
scores["Average_Precision"].append(cv_result["test_average_precision"].mean())
scores["Recall"].append(cv_result["test_recall"].mean())

df_output = pd.DataFrame(scores, index=index)
df_output

## Support Vector Machine (tot nu toe beste op testset)
from sklearn.svm import LinearSVC
model_svc = LinearSVC(random_state=42, dual = False, max_iter=2000)
cv_result = cross_validate(model_svc, X_train, y_train, scoring=scoring)

index += ["Support Vector Machine"]
scores["Precision"].append(cv_result["test_precision"].mean())
scores["Average_Precision"].append(cv_result["test_average_precision"].mean())
scores["Recall"].append(cv_result["test_recall"].mean())

df_output = pd.DataFrame(scores, index=index)
df_output

## Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
model_dt = DecisionTreeClassifier(random_state=42)
cv_result = cross_validate(model_dt, X_train, y_train, scoring=scoring)

index += ["Decision Tree Classifier"]
scores["Precision"].append(cv_result["test_precision"].mean())
scores["Average_Precision"].append(cv_result["test_average_precision"].mean())
scores["Recall"].append(cv_result["test_recall"].mean())

df_output = pd.DataFrame(scores, index=index)
df_output

## Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
model_sgd = SGDClassifier(random_state=42)
cv_result = cross_validate(model_sgd, X_train, y_train, scoring=scoring)

index += ["Stochastic Gradient Descent"]
scores["Precision"].append(cv_result["test_precision"].mean())
scores["Average_Precision"].append(cv_result["test_average_precision"].mean())
scores["Recall"].append(cv_result["test_recall"].mean())

df_output = pd.DataFrame(scores, index=index)
df_output


## Het beste model nu testen op de test en vali set
## Test model op test set
from sklearn import metrics
model_svc.fit(X_train, y_train)
preds = model_svc.predict(X_test)
precision = metrics.precision_score(y_test, preds)
print(f"Precision =  {(precision * 100).round(1)}") #24.7
print(confusion_matrix(y_test, preds))

## Test model op vali set
preds = model_svc.predict(X_vali)
precision = metrics.precision_score(y_vali, preds)
print(f"Precision =  {(precision * 100).round(1)}") #24.7
print(confusion_matrix(y_vali, preds))

## Undersampling
model_svc.fit(X_res, y_res)
preds = model_svc.predict(X_test)
precision = metrics.precision_score(y_test, preds)
print(f"Precision =  {(precision * 100).round(1)}") #24.7
print(confusion_matrix(y_test, preds))



