import io
import pandas as pd
import requests
import os
if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data_from_api(*args, **kwargs):
    url = 'https://ll.thespacedevs.com/2.0.0/launch/upcoming'
    #C:\python\etl\mage\spellbound\data_loaders
    #../../
    os.makedirs("../data", exist_ok=True)  # mage folder 아래 rocket folder 생성
    response = requests.get(url)
    
    with open("../data/launches.json", "wb") as f:
        f.write(response.content)

    # Assuming the API returns a JSON that can be read into a DataFrame
    data = response.json()

    return pd.DataFrame(data)


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'