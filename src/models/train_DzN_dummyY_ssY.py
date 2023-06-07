#Train model
import pandas as pd
import numpy as np

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

## Feature selection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, make_scorer, average_precision_score 
from sklearn.model_selection import learning_curve, StratifiedKFold, train_test_split
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from eli5.sklearn import PermutationImportance
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import Counter

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Plots a learning curve. http://scikit-learn.org/stable/modules/learning_curve.html
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

dict_characters = {0: 'Positief_advies', 1: 'Negatief_advies'}

def runRandomForest(a, b, c, d):
    model = RandomForestClassifier(n_estimators=100, random_state=None, class_weight='balanced_subsample')
    precision_scorer = make_scorer(average_precision_score)
    model.fit(a, b)
    kfold = model_selection.KFold(n_splits=10, random_state=None)
    precision = model_selection.cross_val_score(model, c, d, cv=kfold, scoring='precision')
    mean = precision.mean() 
    stdev = precision.std()
    prediction = model.predict(c)
    cnf_matrix = confusion_matrix(d, prediction)
    #plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
    plot_learning_curve(model, 'Learning Curve For RandomForestClassifier', a, b, (0.80,1.1), 10)
    plt.show()
    plot_confusion_matrix(cnf_matrix, classes=dict_characters,title='Confusion matrix')
    plt.show()
    print('Random Forest Classifier - Training set accuracy: %s (%s)' % (mean, stdev))
    return
runRandomForest(X_train, y_train, X_test, y_test)

model = RandomForestClassifier(n_estimators=100, random_state=None, class_weight='balanced_subsample')
model.fit(X_train,y_train)
columns = X_train.columns
coefficients = model.feature_importances_.reshape(X_train.columns.shape[0], 1)
absCoefficients = abs(coefficients)
fullList = pd.concat((pd.DataFrame(columns, columns = ['Variable']), pd.DataFrame(absCoefficients, columns = ['absCoefficient'])), axis = 1).sort_values(by='absCoefficient', ascending = False)
print('RandomForestClassifier - Feature Importance:')
print('\n',fullList,'\n')

X_train=np.asarray(X_train)
X_test=np.asarray(X_test)
Y_train=np.asarray(y_train)
Y_test=np.asarray(y_test)

# BEGIN: FEATURE SELECTION WITH PERMUTATION IMPORTANCE METHOD
#http://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html
sel = SelectFromModel(PermutationImportance(RandomForestClassifier(n_estimators=10, random_state=None, class_weight='balanced_subsample'), cv=5),threshold=0.005,).fit(X_train, Y_train)
X_train2 = sel.transform(X_train)
X_test2 = sel.transform(X_test)
# END: FEATURE SELECTION WITH PERMUTATION IMPORTANCE METHOD

runRandomForest(X_train2, Y_train, X_test2, Y_test)


## Naive classifier
from sklearn.dummy import DummyClassifier
model_naive = DummyClassifier(strategy='stratified') #random class
model_naive.fit(X_train, y_train)
yhat = model_naive.predict(X_train)

from sklearn.metrics import precision_score
precision = precision_score(y_train, yhat)
print(precision) # 0.16 precision

## Normale Random Forest
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# define model
model_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced_subsample')
model_rf.fit(X_train, y_train)
preds = model_rf.predict(X_test)

# performance model
kfold = model_selection.KFold(n_splits=10, random_state=None)
precision = model_selection.cross_val_score(model, X_test, y_test, cv=kfold, scoring='precision')
mean = precision.mean()
stdev = precision.std()
prediction = model_rf.predict(X_test)

plot_learning_curve(model_rf, 'Learning Curve For RandomForestClassifier', X_train, y_train, (0.80,1.1), 10)
plt.show()

# Accuracy van model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
scoring = ["precision"]
cv_result = cross_validate(model_rf, X_train, y_train, scoring=scoring)
print(f"Precision: {cv_result['test_precision'].mean():.3f}") # Precision 0, omdat hij nooit een 1 predict. Zie:
set(y_test) - set(preds)

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
ConfusionMatrixDisplay.from_predictions(y_test, preds, display_labels=['Positief_advies', 'Negatief_advies']).plot()
plt.show() 


## Balanced Random Forest
from imblearn.ensemble import BalancedRandomForestClassifier
brf = BalancedRandomForestClassifier(max_depth=None, n_estimators=100,random_state=42, class_weight='balanced')
brf.fit(X_train, y_train)
preds_brf = brf.predict(X_test)
precision = metrics.precision_score(y_test, preds_brf)
print(f"Precision =  {(precision * 100).round(1)}") #24.6

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
ConfusionMatrixDisplay.from_predictions(y_test, preds_brf, display_labels=['Positief_advies', 'Negatief_advies']).plot()
plt.show()

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
scoring = ["precision"]
cv_result = cross_validate(brf, X_train, y_train, scoring=scoring)
print(f"Precision: {cv_result['test_precision'].mean():.3f}")

## Permutatie importance (aangezien var importance van RF prioriteit geeft aan numerieke data)
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
perm_importance = permutation_importance(brf, X_test, y_test, n_repeats=5, random_state=42, scoring='precision')

sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(X_train.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.show()

out = pd.DataFrame(X_train.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])

out

out.to_csv("c:/users/dave/desktop/temp.csv")



######################################################################
## Parameter tuning voor BRF
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn import metrics

brf = BalancedRandomForestClassifier(random_state=42)
brf.fit(X_train, y_train)
preds_brf = brf.predict(X_test)
precision = metrics.precision_score(y_test, preds_brf)
print(f"Precision =  {(precision * 100).round(1)}") #24.0%

from sklearn.model_selection import GridSearchCV

params = { 
    'n_estimators': [100, 200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

CV_brf = GridSearchCV(brf, param_grid=params, cv = 5, scoring='precision')
CV_brf.fit(X_train, y_train)
CV_brf.best_params_ #{'criterion': 'entropy', 'max_depth': 7, 'max_features': 'auto', 'n_estimators': 100}

brf2 = BalancedRandomForestClassifier(criterion='entropy', max_depth=7, max_features='auto', n_estimators=100,random_state=42, class_weight='balanced_subsample')
brf2.fit(X_train, y_train)
preds = brf2.predict(X_test)

threshold = 0.8
preds_prob = brf2.predict_proba(X_test)
preds = (preds_prob[:,1] >= threshold).astype('int')

precision = metrics.precision_score(y_test, preds)
print(f"Precision =  {(precision * 100).round(1)}") #26.5%


ConfusionMatrixDisplay.from_predictions(y_test, preds, display_labels=['Positief_advies', 'Negatief_advies']).plot() #896 goede negatief
plt.show()


## Parameter tuning voor normale RF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold

rf = RandomForestClassifier(random_state=42, class_weight='balanced', max_features='log2')

params = { 
    'n_estimators': [10, 20, 50],
}

CV_rf = GridSearchCV(rf, param_grid=params, cv = 10, scoring='precision')
CV_rf.fit(X_train, y_train) ## duurt te lang
CV_rf.best_params_



rf = RandomForestClassifier(random_state=42, n_estimators=200, criterion='gini', class_weight='balanced', max_features='sqrt')
rf.fit(X_train, y_train)
preds = rf.predict(X_test)
precision = metrics.precision_score(y_test, preds)
print(f"Precision =  {(precision * 100).round(1)}") #59.9 nu met laatste cols eraf

threshold = 0.65
preds_prob = rf.predict_proba(X_test)
preds = (preds_prob[:,1] >= threshold).astype('int')
precision = metrics.precision_score(y_test, preds)
print(f"Precision =  {(precision * 100).round(1)}")
ConfusionMatrixDisplay.from_predictions(y_test, preds, display_labels=['Positief_advies', 'Negatief_advies']).plot()
plt.show()


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
scoring = ["precision"]
cv_result = cross_validate(rf, X_train, y_train, scoring=scoring)
print(f"Precision: {cv_result['test_precision'].mean():.3f}")



### Log regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=42, class_weight={0 : 0.5, 1 : 0.5})
lr.fit(X_train, y_train)
preds = lr.predict(X_test)
precision = metrics.precision_score(y_test, preds)
print(f"Precision =  {(precision * 100).round(1)}") #54.4
ConfusionMatrixDisplay.from_predictions(y_test, preds, display_labels=['Positief_advies', 'Negatief_advies']).plot()
plt.show()



### er is nog een optie om met thresholds te werken: https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/


## Imblearn classifiers:
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier, EasyEnsembleClassifier, RUSBoostClassifier
from sklearn.metrics import confusion_matrix

bfc = BalancedRandomForestClassifier(random_state=42)
bfc.fit(X_train, y_train)
preds = bfc.predict(X_test)
precision = metrics.precision_score(y_test, preds)
print(f"Precision =  {(precision * 100).round(1)}") #24
print(confusion_matrix(y_test, preds))

bbc = BalancedBaggingClassifier(random_state=42)
bbc.fit(X_train, y_train)
preds = bbc.predict(X_test)
precision = metrics.precision_score(y_test, preds)
print(f"Precision =  {(precision * 100).round(1)}") #24.7
print(confusion_matrix(y_test, preds))

from sklearn.ensemble import HistGradientBoostingClassifier
eec = EasyEnsembleClassifier(random_state=42, sampling_strategy='all', estimator=HistGradientBoostingClassifier(random_state=42))
eec.fit(X_train, y_train)
preds = eec.predict(X_test)
precision = metrics.precision_score(y_test, preds)
print(f"Precision =  {(precision * 100).round(1)}") #25.8 deze is het beste
print(confusion_matrix(y_test, preds))


rbc = RUSBoostClassifier(random_state=42)
rbc.fit(X_train, y_train)
preds = rbc.predict(X_test)
precision = metrics.precision_score(y_test, preds)
print(f"Precision =  {(precision * 100).round(1)}") #25.7
print(confusion_matrix(y_test, preds))

## Tabel maken voor alle preds
import pandas as pd

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