import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

STATE = 516841

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('landslide')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


ls_cdf = pd.read_csv("../data/nasa_global_landslide_catalog_point.csv")
fake_rows = ls_cdf.sample(2000)
fake_rows['landslide'] = 0
ls_cdf['landslide'] = 1

# randomly_vary X for the fake rows
fake_rows['gazetteer_distance'] = fake_rows.gazetteer_distance.apply(lambda x: x * np.random.uniform(0.8, 1.2))

# stack vertically
dataframe = pd.concat(
    [ls_cdf[['gazetteer_distance', 'landslide']], 
    fake_rows[['gazetteer_distance', 'landslide']]], 
    ignore_index=True).reset_index(drop=True)

# output toy data
dataframe.to_pickle('../data/toy_data.pkl')

train, test = train_test_split(dataframe, test_size = .2, random_state = STATE)
train, val = train_test_split(train, test_size=.2, random_state = STATE)

print(len(train), 'train examples')
print(len(val), 'val examples')
print(len(test), 'test examples')

# batch_size = 5 
# train_ds = df_to_dataset(train, batch_size=batch_size)
# val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
# test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# for feature_batch, label_batch in train_ds.take(1):
#     print('Every feature:', list(feature_batch.keys()))
#     print('ages', feature_batch['gazetteer_distance'])
#     print('targets', label_batch)

# example_batch = next(iter(train_ds))
feature_columns = []
for header in ['gazetteer_distance']:
    feature_columns.append(tf.feature_column.numeric_column(header))

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

for feature_batch, label_batch in train_ds.take(1):
    print('Every feature:', list(feature_batch.keys()))
    print('ages', feature_batch['gazetteer_distance'])
    print('targets', label_batch)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
    feature_layer,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(.1),
    tf.keras.layers.Dense(1)
])

## For multi class classification, loss is done via categorical cross entropy instead of binary.
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=10)


# arr = np.array(merged_cdf)
# arr_np = np.asarray(arr)
# np_split = np.hsplit(arr_np,2)
# input_vector = np_split[0]
# output_vector = np_split[1]
# print(input_vector.shape)
# print(output_vector.shape)pi