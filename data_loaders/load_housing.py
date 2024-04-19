from sklearn.datasets import fetch_california_housing
from pandas import DataFrame
import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

print(fetch_california_housing().keys())

@data_loader
def load_data_from_api(**kwargs) -> DataFrame:
    X = pd.DataFrame(fetch_california_housing().data,
                 columns=fetch_california_housing().feature_names)
    y = pd.DataFrame(fetch_california_housing().target,
                 columns=fetch_california_housing().target_names)
    return pd.merge(X, y, left_index=True, right_index=True)


@test
def test_output(df) -> None:
    """
    Template code for testing the output of the block.
    """
    assert df is not None, 'The output is undefined'