import numpy as np
import os
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------
# データ読み込み・拡張
# ---------------------
class GenData():
    def __init__(self, pd_obj):
        self.pd_obj = pd_obj

    def load_npdata(self, data_dir_path):
        file_list = os.listdir(data_dir_path)
        data_list = []
        for category_index, file in enumerate(file_list):
            array = np.load(os.path.join(data_dir_path, file))
            for data in array:
                data_list.append(data)

        return data_list






if __name__ == '__main__':
    root_dir_path = r"C:\Users\user1\Desktop\oshima\scene_sketch_recognition"
    img_dir_path = os.path.join(root_dir_path, r"data\interim\scene_sketch225u")
    csv_file_path = os.path.join(root_dir_path, r"data\label\scene_sketch_label.csv")

    df = pd.read_csv(csv_file_path)
    data = GenData(df)
    img_data = data.load_npdata(img_dir_path)

    print(df)