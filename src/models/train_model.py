#Train model
import pandas as pd


# Read data
df = pd.read_parquet("C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/processed/output_dataset.parquet")


df = df.replace('positief_advies', 0)
df = df.replace('negatief_advies', 1)

def type_fixer(df):
    ## alle types naar de goede format zetten
    
    uint8 = list(df.select_dtypes('uint8').columns)
    uint16 = list(df.select_dtypes('uint16').columns)
    
    to_cat2 = list(df.columns[df.columns.str.contains("t0")])
    numericals = ['t0_eq5d_index', 't0_eq_vas', 'oks_t0_score']
    to_cat2 = [element for element in to_cat2 if element not in numericals] 
    naar_cat = uint8 + uint16 + to_cat2
    
    for col in naar_cat:
        df[col] = df[col].astype('category')
    
    return df

df = type_fixer(df)


## Train test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(['succesfaction_or', 'succesfaction_and'], axis=1), df['succesfaction_or'], test_size=0.33, random_state=42, stratify=df['succesfaction_or'])

## One Hot Encoding
cats = list(X_train.select_dtypes('category').columns)
X_train = pd.get_dummies(X_train, columns=cats)

cats = list(X_test.select_dtypes('category').columns)
X_test = pd.get_dummies(X_test, columns=cats)

## Naive classifier
from sklearn.dummy import DummyClassifier
model_naive = DummyClassifier(strategy='stratified') #random class
model_naive.fit(X_train, y_train)
yhat = model_naive.predict(X_train)

from sklearn.metrics import precision_score
precision = precision_score(y_train, yhat)
print(precision) # 0.02 precision

## Normale Random Forest
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# define model
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
preds = model_rf.predict(X_test)

# Accuracy van model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scoring = ["precision"]
cv_result = cross_validate(model_rf, X_train, y_train, scoring=scoring)
print(f"Precision: {cv_result['test_precision'].mean():.3f}") # Precision 0, omdat hij nooit een 1 predict. Zie:
set(y_test) - set(preds)

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
out = ConfusionMatrixDisplay.from_predictions(y_test, preds, display_labels=['Positief_advies', 'Negatief_advies']).plot()
plt.show() 


## Balanced Random Forest
from imblearn.ensemble import BalancedRandomForestClassifier
brf = BalancedRandomForestClassifier(max_depth=None, n_estimators=10,random_state=42, class_weight=None)
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