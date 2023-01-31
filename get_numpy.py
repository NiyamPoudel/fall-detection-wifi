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
final_array = []

def process(data):
    print(data.shape)

if __name__ == "__main__":
    FILE_NAME = "/home/nimai/esp/analyse/data/exp55.csv"
    f = open(FILE_NAME)
    for j, l in enumerate(f.readlines()):
        res = l
        imaginary = []
        real = []
        amplitudes = []
        tuple_amp = []
        phases = []
        x = []
        all_data = res.split(",")
        # print(all_data[27].strip())
        csi_data = all_data[25].split(" ")
        csi_data[0] = csi_data[0].replace("[", "")
        csi_data[-1] = csi_data[-1].replace("]", "")
        csi_data.pop()
        # print(csi_data)
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
            # print("-------------------")
            # print("csi_amplitude#",np.array(amplitudes).shape)
            # from CSIKit.filters.wavelets.dwt import denoise
            # finalEntry = denoise(np.array(amplitudes))
            # print(finalEntry)
            for idx, val in enumerate(amplitudes):
                temp_arr = []
                temp_arr.append(idx)
                temp_arr.append(val)
                tup = temp_arr
                tuple_amp.append((tup))
            
            tuple_amp.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            tuple_amp.append(all_data[27].strip())
            final_array.append(tuple_amp)
    final_numpy = np.array(final_array)
    print("final shape",final_numpy.shape)
    print(final_numpy.shape)
    np.save("exp55", final_numpy)
