import pandas as pd
import numpy as np

quakes = pd.read_csv('input/3.5+Earthquakes_CA_since_2000.csv')
slides = pd.read_csv('input/nasa_global_landslide_catalog_point.csv')

print(slides.shape)
print(slides.landslide_trigger.value_counts())
# slides_q = slides.loc[slides.landslide_trigger == 'earthquake'])

