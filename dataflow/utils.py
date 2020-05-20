import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

RESOLUTION = 256  # Ideally we shouldn't be resizing but I'm lacking memory


def compute_std_avg(data_path):
    data = []
    df_train = pd.read_csv(os.path.join(data_path, 'train.csv'))

    for file in tqdm(df_train['Images'], miniters=256):
        img_path = os.path.join(data_path, 'data', file)

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            data.append(cv2.resize(img, (RESOLUTION, RESOLUTION)))

    data = np.array(data, np.float32) / 255
    print("Shape: ", data.shape)

    means = []
    stdevs = []
    for i in range(3):
        pixels = data[:, :, :, i].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    print("means: {}".format(means))
    print("stdevs: {}".format(stdevs))
    print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))

    return means, stdevs

##############TRAIN################################################
# Shape:  (44042, 256, 256, 3)
# means: [0.57268673, 0.57268673, 0.57268673]
# stdevs: [0.23658775, 0.23658775, 0.23658775]
# transforms.Normalize(mean = [0.57268673, 0.57268673, 0.57268673], std = [0.23658775, 0.23658775, 0.23658775])
##############VALID&TEST##################################
# Shape:  (1545, 256, 256, 3)
# means: [0.5769622, 0.5769622, 0.5769622]
# stdevs: [0.23130997, 0.23130997, 0.23130997]
# transforms.Normalize(mean = [0.5769622, 0.5769622, 0.5769622], std = [0.23130997, 0.23130997, 0.23130997])
