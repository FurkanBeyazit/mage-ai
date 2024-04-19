import os

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def notify(*args, **kwargs):
    images_count = len(os.listdir("../image"))
    
    if images_count > 0:
        print(f"There are now {images_count} images.")
    else :
        print("No images downloaded")


