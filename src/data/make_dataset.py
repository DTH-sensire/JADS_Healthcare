# -*- coding: utf-8 -*-
import os
import click
import logging
import pandas as pd
import numpy as np


input_filepath = r"C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/raw/knee-provider.parquet"
output_filepath = r"C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/processed/output_dataset.parquet"
df = pd.read_parquet(input_filepath)


def only_interesting(df):
    ## Hier haal ik alle t1 vars weg die we niet gebruiken
    t1_list = list(df.columns[df.columns.str.contains("t1")])
    keep = ["t1_satisfaction", "t1_sucess", "oks_t1_score"] #deze hebben we nog wel nodig
    t1_list = [element for element in t1_list if element not in keep] 
    
    t1_list.extend(["provider_code", 't0_eq5d_index_profile', 't1_eq5d_index_profile']) #niet relevant door dataset
    
    return df.drop(t1_list, axis=1) 


def DeventerZiekenhuis(df):
    ## Volgende cols weg vanwege Deventer Ziekenhuis limits
    dropped = ['heart_disease', 'high_bp', 'stroke', 'circulation', 'lung_disease', 'diabetes', 'kidney_disease', 'nervous_system', 'liver_disease', 'cancer', 'depression', 'arthritis', 't0_assisted', 't0_symptom_period', 't0_previous_surgery', 't0_living_arrangements', 't0_disability', 'revision_flag']
    return df.drop(dropped, axis=1)


def missing_values(df):
    ## Missing values worden hier verwijderd of aangepast
    
    ## Eerst de 9 omzetten naar 0
    if dz_input == 'N':
        naar_zero = ['heart_disease', 'high_bp', 'stroke', 'circulation', 'lung_disease', 'diabetes', 'kidney_disease', 'nervous_system', 'liver_disease', 'cancer', 'depression', 'arthritis']
        df[naar_zero] = df[naar_zero].replace(9, 0)
    
    a = list(df.columns)
    keep2 = ['oks_t0_score', 'oks_t1_score',"t0_eq_vas"]
    a = [element for element in a if element not in keep2] 
    
    # VAS alleen pakken
    df['t0_eq_vas'] = df['t0_eq_vas'].replace(999,np.nan)
    
    #De rest naar nan
    df[a] = df[a].replace(9, np.nan)
    
    initial_size = len(df)
    df = df.dropna(axis=0, how = 'any') #alles verwijderen
    after_size = len(df)
    
    print(f"Er zijn {initial_size - after_size} observaties verwijderd. Dit is {(initial_size - after_size) / initial_size * 100}%")
    
    return df


def near_zero_variances(df):
    ## Eerst de numerical
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold() #threshold van 0
    selector.fit(df.select_dtypes('number'))
    out = selector.get_support()
    out2 = [not elem for elem in out]
    vars_drop = list(df.select_dtypes('number').columns[out2])
        
    # String vars toevoegen
    vars_drop.append("procedure") #deze string column bevat alleen dezelfde strings
    
    print(f"{vars_drop} zijn verwijderd")
    
    # Verwijderen columns
    return df.drop(vars_drop, axis=1)


def nieuwe_vars(df):
    ## Hier worden nieuwe variabelen aangemaakt
    df['oks_change_score'] = df.oks_t1_score - df.oks_t0_score
    df['oks_MID_7'] = np.where((df.oks_change_score >= 7), 'CHANGE','NO_CHANGE') 
    df['succesfaction_and'] = np.where((df.t1_sucess > 3) & (df.t1_satisfaction > 4) & (df.oks_MID_7 == 'NO_CHANGE'), 'negatief_advies', 'positief_advies')
    df['succesfaction_or'] = np.where((df.t1_sucess > 3) | (df.t1_satisfaction > 4) | (df.oks_MID_7 == 'NO_CHANGE'), 'negatief_advies', 'positief_advies')
    
    return df


def type_fixer(df):
    ## alle types naar de goede format zetten
    
    if dummy_input == 'Y':
        uint8 = list(df.select_dtypes('uint8').columns)
        uint16 = list(df.select_dtypes('uint16').columns)
        
        to_cat2 = list(df.columns[df.columns.str.contains("t0")])
        numericals = ['t0_eq5d_index', 't0_eq_vas', 'oks_t0_score']
        to_cat2 = [element for element in to_cat2 if element not in numericals] 
        naar_cat = uint8 + uint16 + to_cat2
        
        for col in naar_cat:
            df[col] = df[col].astype('category')
    
    df['gender'] = df['gender'].astype('bool')
    
    return df


def final_cleaning(df):
    t1_list = list(df.columns[df.columns.str.contains("t1")])
    t1_list.extend(['oks_change_score', 'oks_MID_7'])
    df = df.drop(t1_list, axis=1)
    
    df = df.replace('geslaagde_behandeluitkomst', 0)
    df = df.replace('nietgeslaagde_behandeluitkomst', 1)
    
    return df

def train_test_vali_split(df):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['succesfaction_or', 'succesfaction_and'], axis=1), df['succesfaction_or'], test_size=0.10, random_state=42, stratify=df['succesfaction_or'])
    X_train, X_vali, y_train, y_vali = train_test_split(X_train, y_train, test_size=0.10, random_state=42, stratify=y_train)
    
    return(X_train, X_test, X_vali, y_train, y_test, y_vali)

def onehotEncode(X_train, X_test, X_vali):
    cats = list(X_train.select_dtypes('category').columns)
    X_train = pd.get_dummies(X_train, columns=cats)
    X_test = pd.get_dummies(X_test, columns=cats)
    X_vali = pd.get_dummies(X_vali, columns=cats)
    
    return(X_train, X_test, X_vali)

def standardScaling(X_train, X_test, X_vali):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    nums = list(X_train.select_dtypes(['float32', 'float64']).columns)
    X_train[nums] = scaler.fit_transform(X_train[nums])
    X_test[nums] = scaler.fit_transform(X_test[nums])
    X_vali[nums] = scaler.fit_transform(X_vali[nums])
    
    return(X_train, X_test, X_vali)

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

def main(input_filepath, output_filepath):

    df = pd.read_parquet(input_filepath)
    df = only_interesting(df)
    
    dz_input = input("Dataset voor Deventer Ziekenhuis: (Y/N) ")
    if dz_input == "Y":
        df = DeventerZiekenhuis(df)
    
    df = missing_values(df)
    df = near_zero_variances(df)
    df = nieuwe_vars(df)
    
    dummy_input = input("Dummy coderen?: (Y/N) ")
    df = type_fixer(df)
    
    df = final_cleaning(df)
    X_train, X_test, X_vali, y_train, y_test, y_vali = train_test_vali_split(df)
    X_train, X_test, X_vali = onehotEncode(X_train, X_test, X_vali)
    
    ss_input = input("Standard Scaling?: (Y/N) ")
    if ss_input == 'Y':
        X_train, X_test, X_vali = standardScaling(X_train, X_test, X_vali)
    
    out = r"C:/Users/Dave/Desktop/JADS/JADS_project/JADS_Healthcare/data/processed/"
    
    X_train.to_parquet(f"{out}X_train_Dz{dz_input}_Dummy{dummy_input}_Ss{ss_input}.parquet")
    X_test.to_parquet(f"{out}X_test_Dz{dz_input}_Dummy{dummy_input}_Ss{ss_input}.parquet")
    X_vali.to_parquet(f"{out}X_vali_Dz{dz_input}_Dummy{dummy_input}_Ss{ss_input}.parquet")
    y_train.to_csv(f"{out}y_train_Dz{dz_input}_Dummy{dummy_input}_Ss{ss_input}.csv", index=False)
    y_test.to_csv(f"{out}y_test_Dz{dz_input}_Dummy{dummy_input}_Ss{ss_input}.csv", index=False)
    y_vali.to_csv(f"{out}y_vali_Dz{dz_input}_Dummy{dummy_input}_Ss{ss_input}.csv", index=False)
    
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
    
main(input_filepath=input_filepath, output_filepath=output_filepath)