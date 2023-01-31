"""
"""

import time
import math
from datetime import datetime, timedelta
import numpy as np
import tensorflow as tf
import cv2


# model = tf.keras.models.load_model('/home/nimai/workspace/fall-detection-wifi/Models/dataset_22.h5py')
# file_path = "/home/nimai/workspace/fall-detection-wifi/Videos/exp_64_l_2.mp4"
# FILE_NAME = "/home/nimai/workspace/fall-detection-wifi/csv/exp64_l_2.csv"

model = tf.keras.models.load_model('/home/nimai/workspace/fall-detection-wifi/Models/dataset_25.h5py')
file_path = "/home/nimai/workspace/fall-detection-wifi/Videos/exp65_l_2.mp4"
FILE_NAME = "/home/nimai/workspace/fall-detection-wifi/csv/exp65_l_2.csv"
f = open(FILE_NAME)
next(f)
data = f.readlines()
streamer = cv2.VideoCapture(file_path)
fps = streamer.get(cv2.CAP_PROP_FPS)
length = int(streamer.get(cv2.CAP_PROP_FRAME_COUNT))
# date = datetime.strptime("2021-09-01 14:44:29", "%Y-%m-%d %H:%M:%S")
date = datetime.strptime("2021-09-02 14:04:15", "%Y-%m-%d %H:%M:%S")
batch_size = 40
idx = 0
arr_to_predict = []
while True:
    ret, frame = streamer.read()
    frame = cv2.resize(frame, (640, 360))
    date = date + timedelta(seconds=(1./fps))
    wifi_date = data[idx].split(",")[-2]
    while datetime.strptime(date.strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S") > datetime.strptime(wifi_date, "%Y-%m-%d %H:%M:%S"):
        idx += 1
        wifi_date = data[idx].split(",")[-2]
    if str(date.strftime("%Y-%m-%d %H:%M:%S")) == str(wifi_date):
        csi_data = data[idx].split(",")[25].replace("[", "").replace("]", "").split(" ")
        csi_data.pop()
        csi_data = [int(c) for c in csi_data]
        csi_data = csi_data[12:]
        csi_data = csi_data[: len(csi_data) - 10]
        imaginary = []
        real = []
        amplitudes = []
        for i in range(len(csi_data)):
            if i % 2 == 0:
                imaginary.append(csi_data[i])
            else:
                real.append(csi_data[i])
        for i in range(int(len(csi_data) / 2)):
            amplitudes.append(math.sqrt(imaginary[i] ** 2 + real[i] ** 2))
        arr_to_predict.append(amplitudes)
        idx += 1
    
    if len(arr_to_predict) == (batch_size + 1):
        arr_to_predict.pop(0)
        arr_np = np.array(arr_to_predict)
        arr_np = arr_np.reshape(-1, batch_size, 53, 1)
        arr_np = arr_np.astype('float32')
        arr_np = arr_np / 255.
        output = model.predict(arr_np)
        output_idx = np.argmax(output[0])
        if output_idx == 0:
            print("No fall")
            # frame = cv2.putText(frame, "No fall", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.1, (255, 0, 0), 1)
        else:
            print("Fall Detected")
            # frame = cv2.putText(frame, "Fall", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.1, (255, 0, 0), 1)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
streamer.release()