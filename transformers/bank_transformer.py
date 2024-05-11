from pandas import DataFrame
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from mage_ai.data_preparation.decorators import data_loader

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader



def preprocess_dataframe(df):
    # Handling outliers
    if 'balance' in df.columns:
        q1 = df['balance'].quantile(0.25)
        q3 = df['balance'].quantile(0.75)
        iqr = q3 - q1
        df['balance'] = df['balance'].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
    if 'campaign' in df.columns:
        df['campaign'] = df['campaign'].clip(upper=df['campaign'].quantile(0.99))
    if 'previous' in df.columns:
        df['previous'] = df['previous'].clip(upper=df['previous'].quantile(0.99))

    # Dropping columns
    columns_to_drop = ['poutcome', 'default', 'day', 'month']
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True, axis=1)
    
    return df
def dummies(df):
    df= pd.get_dummies(df)
    df=df.replace(True,1)
    df=df.replace(False,0)
    return df
def scale(df):
    numerical_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df
def target_scale(df):
    df['target']=df['target'].replace('no',0) 
    df['target']=df['target'].replace('yes',1)    
    return df


@transformer
def transform_df(df: DataFrame, *args, **kwargs) -> DataFrame:
     return scale(dummies(target_scale(preprocess_dataframe(df))))
    


@test
def test_output(df) -> None:
    """
    Template code for testing the output of the block.
    """
    assert df is not None, 'The output is undefined'