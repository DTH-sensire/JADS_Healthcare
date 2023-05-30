import pandas as pd
import numpy as np

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
X_train, X_test, y_train, y_test = train_test_split(df.drop(['succesfaction_or', 'succesfaction_and'], axis=1), df['succesfaction_or'], test_size=0.10, random_state=42, stratify=df['succesfaction_or'])
X_train, X_vali, y_train, y_vali = train_test_split(X_train, y_train, test_size=0.10, random_state=42, stratify=y_train)


## One Hot Encoding
cats = list(X_train.select_dtypes('category').columns)
X_train = pd.get_dummies(X_train, columns=cats)

cats = list(X_test.select_dtypes('category').columns)
X_test = pd.get_dummies(X_test, columns=cats)

cats = list(X_vali.select_dtypes('category').columns)
X_vali = pd.get_dummies(X_vali, columns=cats)



## Model instance om mee te testen
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10, random_state=None, class_weight='balanced_subsample')

## Univariate feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

BestFeatures = SelectKBest(score_func=chi2, k=10)
fit = BestFeatures.fit(X_train.drop(["t0_eq5d_index"], axis=1),y_train) #moet wel zonder eq5d_index

df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X_train.drop(["t0_eq5d_index"], axis=1).columns)
f_Scores = pd.concat([df_columns,df_scores],axis=1)
f_Scores.columns = ['Specs','Score']
out = f_Scores.nlargest(10,'Score')

out.to_csv("c:/users/dave/desktop/temp.csv")

y_train[X_train[X_train['oks_t0_limping_4.0'] == True].index].value_counts(normalize=True)


## Wrapper methods
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

sfs1 = SFS(model, 
           k_features=10, 
           forward=True, 
           floating=False, 
           verbose=2,
           scoring='r2',
           cv=3)

sfs1 = sfs1.fit(np.array(X_train), y_train)

X_train.columns[list(sfs1.k_feature_idx_)]


