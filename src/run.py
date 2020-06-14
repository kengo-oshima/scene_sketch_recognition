import os
import pandas as pd
from src.data.generate_data import GenData
from pprint import pprint

def generate_id(csv_file_path):


def main():
    root_dir_path = r'C:\Users\user1\Desktop\oshima\scene_sketch_recognition'
    root_dir_path = r"C:\Users\user1\Desktop\oshima\scene_sketch_recognition"
    img_dir_path = os.path.join(root_dir_path, r"data\interim\scene_sketch225u")
    csv_file_path = os.path.join(root_dir_path, r"data\label\scene_sketch_label.csv")

    df = pd.read_csv(csv_file_path)
    data = GenData(df)
    img_data = data.load_npdata(img_dir_path)
    label = list(data.load_label())

    pprint(label)

if __name__ == '__main__':
    main()