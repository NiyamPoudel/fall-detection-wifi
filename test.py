import os
import numpy as np

import json
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split


batch_size = 40
npy_path = "/home/nimai/workspace/fall-detection-wifi/Numpys/exp65_l_2.npy"
data = np.load(npy_path, allow_pickle=True)
labels = data[:, -1]
timestamp = data[:, -2]
data = np.delete(data, -1, 1)
data = data.astype(np.float)
x_dataset = []
y_label = []
for idx, d in enumerate(data):
    dd = data[idx:idx+batch_size]
    label = labels[idx:idx+batch_size]
    label = label[np.argmax(label)]
    x, y = dd.shape
    if x == batch_size:

        x_dataset.append(dd)
        if label == 'nofall':
            y_label.append(0)
        else:
            y_label.append(1)

x_dataset, y_label= np.array(x_dataset), np.array(y_label)

print(x_dataset.shape, y_label.shape)
x_dataset = x_dataset.reshape(-1, batch_size,53, 1)
x_dataset = x_dataset.astype('float32')
x_dataset = x_dataset / 255.
y_label = to_categorical(y_label)
model = tf.keras.models.load_model('/home/nimai/workspace/fall-detection-wifi/Models/dataset_25.h5py')
model.summary()

labels = ["nofall", "fall"]
# print(x_dataset.shape)
label_arr = []
for idx, data in enumerate(x_dataset):
    label = y_label[idx]
    predictions = model.predict(np.array([data]))
    label_arr.append(label)
    print("GT: ", labels[np.argmax(label)], idx)
    print("DT: ", labels[np.argmax(predictions)])

# print(label_arr)