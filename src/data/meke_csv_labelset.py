import os
import numpy as np
import pandas as pd

class ImgArray():
    def __init__(self, array, array_num):
        self.img_array = array
        self.environment = None
        self.category_label = None
        self.environment = None
        self.img_num = array_num

def load_npdata(data_dir_path):
    file_list = os.listdir(data_dir_path)
    data_list = []
    env_list = [1, 1, 0, 1, 0, 1, 0, 1, 1, 0]
    for category_index, file in enumerate(file_list):
        array = np.load(os.path.join(data_dir_path, file))
        for i in range(array.shape[0]):
            img_data = ImgArray(array[i], i)
            img_data.category_label = category_index
            img_data.environment = env_list[category_index]
            data_list.append(img_data)

    return data_list

def main():
    root_dir_path = r"C:\Users\user1\Desktop\oshima\scene_sketch_recognition"
    img_dir_path = os.path.join(root_dir_path, r"data\interim\scene_sketch225u")
    csv_file_path = os.path.join(root_dir_path, r"data\label\scene_sketch_label.csv")
    csv_obj_list = load_npdata(img_dir_path)
    label_list = []
    for csv_obj in csv_obj_list:
        label_list.append([csv_obj.category_label, csv_obj.environment, csv_obj.img_num])
    df = pd.DataFrame(label_list, columns=['category', 'environment', 'num par category'])
    df = pd.DataFrame(df)
    df.to_csv(csv_file_path)

    print(df)

if __name__ == '__main__':
    main()