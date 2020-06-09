import numpy as np
import os
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# ---------------------
# データ読み込み・拡張
# ---------------------
class GenData():
    def __init__(self, pd_obj):
        self.pd_obj = pd_obj







if __name__ == '__main__':
    root_dir_path = r"C:\Users\user1\Desktop\oshima\scene_sketch_recognition"
    img_dir_path = os.path.join(root_dir_path, r"data\interim\scene_sketch225u")
    csv_file_path = os.path.join(root_dir_path, r"data\label\scene_sketch_label.csv")

    df = pd.read_csv(csv_file_path)

    print(df)