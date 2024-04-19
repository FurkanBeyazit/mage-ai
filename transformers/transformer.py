import json
import requests
import requests.exceptions as requests_exceptions
import os

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@transformer
def get_pictures(data, *args, **kwargs):

    os.makedirs(f"../image", exist_ok=True)
    
    # Explicitly set the encoding to 'utf-8'
    with open("../data/launches.json", encoding='utf-8') as f:
        launches = json.load(f)

    image_urls = [launch["image"] for launch in launches["results"]]

    print(image_urls)

    for image_url in image_urls:
        try:
            response = requests.get(image_url)
            image_filename = image_url.split("/")[-1]
            
            # Fix the target file path
            target_file = os.path.join("../image", image_filename)
            
            with open(target_file, "wb") as f:
                f.write(response.content)

            print(f"Downloaded {image_url} to {target_file}")

        except requests_exceptions.MissingSchema:
            print(f"{image_url} appears to be an invalid URL.")

        except requests_exceptions.ConnectionError:
            print(f"Could not connect to {image_url}.")

    return data

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'