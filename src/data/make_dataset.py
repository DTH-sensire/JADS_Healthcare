# -*- coding: utf-8 -*-
import os
import click
import logging
import pandas as pd
import numpy as np


#input_filepath = "./data/raw/knee-provider.parquet"
#output_filepath = "./data/processed/output_dataset.csv"
#df = pd.read_parquet(input_filepath)


def only_interesting(df):
    ## Hier haal ik alle t1 vars weg die we niet gebruiken
    t1_list = list(df.columns[df.columns.str.contains("t1")])
    keep = ["t1_satisfaction", "t1_sucess", "oks_t1_score"] #deze hebben we nog wel nodig
    t1_list = [element for element in t1_list if element not in keep] 
    
    return df.drop(t1_list, axis=1) 

def type_fixer(df):
    ## alle types naar de goede format zetten
    df = df.provider_code.to_string() #dit is een unieke identifier, dus string
    return df

def missing_values(df):
    ## Missing values worden hier verwijderd of aangepast
    initial_size = len(df)
    df = df.dropna(axis=0, how = 'any') #alles verwijderen
    after_size = len(df)
    
    print(f"Er zijn {initial_size - after_size} observaties verwijderd.")
    
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
    
@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

def main(input_filepath, output_filepath):

    df = pd.read_parquet(input_filepath)
    
    df = only_interesting(df)
    #df = type_fixer(df)
    df = missing_values(df)
    df = near_zero_variances(df)
    df = nieuwe_vars(df)
    
    df.to_csv(output_filepath)
    
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
    
