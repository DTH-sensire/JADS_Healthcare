#Train model
import pandas as pd
import numpy as np

#Load data:
X_train = pd.read_parquet("C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/processed/X_train.parquet")
y_train = pd.read_csv("C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/processed/y_train.csv")
y_train = np.ravel(y_train, order='C')
X_test = pd.read_parquet("C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/processed/X_test.parquet")
y_test = pd.read_csv("C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/processed/y_test.csv")
y_test = np.ravel(y_test, order='C')
X_vali = pd.read_parquet("C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/processed/X_vali.parquet")
y_vali = pd.read_csv("C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/processed/y_vali.csv")
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
ConfusionMatrixDisplay.from_predictions(y_test, prediction, display_labels=['Positief_advies', 'Negatief_advies']).plot()
plt.show() 


## Balanced Random Forest
from imblearn.ensemble import BalancedRandomForestClassifier
brf = BalancedRandomForestClassifier(max_depth=None, n_estimators=10,random_state=42, class_weight='balanced_subsample')
brf.fit(X_train, y_train)
preds_brf = brf.predict(X_test)

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
ConfusionMatrixDisplay.from_predictions(y_test, preds_brf, display_labels=['Positief_advies', 'Negatief_advies']).plot()
plt.show()

from sklearn import metrics
precision = metrics.precision_score(y_test, preds_brf)
print(f"Precision =  {(precision * 100).round(1)}")

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




## Random forest II
from yellowbrick.model_selection import learning_curve

rfc = RandomForestClassifier(n_estimators=10, random_state=42, class_weight='balanced_subsample')
print(learning_curve(rfc, X_train, y_train, cv=10, scoring='precision'))
print("hello")



