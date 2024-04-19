from pandas import DataFrame
import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


def discretinization(df: DataFrame) -> DataFrame:
    df['Longitude']= pd.cut(df['Longitude'], bins= 5)
    df['Latitude'] = pd.cut(df['Latitude'], bins= 5)
    return pd.get_dummies(df)


@transformer
def transform_df(df: DataFrame, *args, **kwargs) -> DataFrame:
    """
    Template code for a transformer block.
    """
    return discretinization(df)
    

@test
def test_output(df) -> None:
    """
    Template code for testing the output of the block.
    """
    assert df is not None, 'The output is undefined'