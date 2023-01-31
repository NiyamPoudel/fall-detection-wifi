import re
from math import sqrt, atan2
import sys
import time
import json
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import numpy as np
import re
import argparse
import csv
from datetime import datetime 
from CSIKit.filters.wavelets.dwt import denoise
final_array = []

def process(data):
    print(data.shape)

if __name__ == "__main__":
    FILE_NAME = "/home/nimai/workspace/fall-detection-wifi/csv/exp67_l.csv"
    f = open(FILE_NAME)
    noised_data = []
    labels = []
    denoised_data_labels = []
    for j, l in enumerate(f.readlines()):
        res = l
        imaginary = []
        real = []
        amplitudes = []
        tuple_amp = []
        phases = []
        x = []
        all_data = res.split(",")
        csi_data = all_data[25].split(" ")
        csi_data[0] = csi_data[0].replace("[", "")
        csi_data[-1] = csi_data[-1].replace("]", "")
        csi_data.pop()
        if len(csi_data) == 128:
            csi_raw = [int(c) for c in csi_data]
            csi_raw = csi_raw[12:]
            csi_raw = csi_raw[: len(csi_raw) - 10]
            for i in range(len(csi_raw)):
                if i % 2 == 0:
                    imaginary.append(csi_raw[i])
                else:
                    real.append(csi_raw[i])
            # Transform imaginary and real into amplitude and phase
            for i in range(int(len(csi_raw) / 2)):
                amplitudes.append(math.sqrt(imaginary[i] ** 2 + real[i] ** 2))
                phases.append(math.atan2(imaginary[i], real[i]))
                x.append(i)
            noised_data.append(np.array(amplitudes))
            labels.append(all_data[27].strip())

    noised_data = np.array(noised_data)
    print(noised_data.shape)
    # denoised_data =  denoise(noised_data)
    denoised_data = noised_data
    denoised_data[np.isnan(denoised_data)] = 0

    print(denoised_data)
    for idx, val in enumerate(denoised_data):
        temp = np.append(val, labels[idx])
        denoised_data_labels.append(temp)

    denoised_data_labels = np.array(denoised_data_labels)
    print(denoised_data_labels.shape)
    np.save("exp67_l", denoised_data_labels)

