import tensorflow as tf
import numpy as np
import pandas as pd
ls_cdf = pd.read_csv("/Users/shaislotky/Desktop/landslide/ML_project/data/nasa_global_landslide_catalog_point.csv")
fake_rows = ls_cdf.sample(1000)
fake_rows['landslide'] = 0
ls_cdf['landslide'] = 1

merged_cdf = pd.concat([ls_cdf[['gazetteer_distance', 'landslide']], fake_rows[['gazetteer_distance', 'landslide']] ])

arr = np.array(merged_cdf)
arr_np = np.asarray(arr)
np_split = np.hsplit(arr_np,2)
input_vector = np_split[0]
output_vector = np_split[1]
print(input_vector.shape)
print(output_vector.shape)