import pandas as pd
import datetime
import os

class ResultCsv:
    def __init__(self, pd_path):
        self.pd_obj = pd.read_csv(pd_path)

    def id_to_index(self, model_id):
        model_index = self.pd_obj[self.pd_obj.id == model_id].index.values
        return int(model_index)

    def generate_id(self):
        id_int = len(self.pd_obj.id)
        model_id = 'M' + str(id_int).zfill(4)
        s = pd.Series([model_id], index=['id'])
        self.pd_obj = self.pd_obj.append(s, ignore_index=True)
        return model_id

    def input_date(self, model_id):
        model_index = self.id_to_index(model_id)
        today = datetime.date.today().strftime('%Y-%m-%d')
        self.pd_obj.loc[model_index, 'date'] = today

    def input_name(self, model_id, model_name):
        model_index = self.id_to_index(model_id)
        self.pd_obj.loc[model_index, 'name'] = model_name

def create_save_dir_path(model_id, model_name):
    root_dir = r'C:\Users\user1\Desktop\oshima\scene_sketch_recognition\results'
    dir_name = model_id + '_' + datetime.date.today().strftime('%Y%m%d')
    dir_path = os.path.join(root_dir, os.path.join(model_name, dir_name))
    create_dir(dir_path)

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(path)
        print("ディレクトリが存在します")
        exit()

if __name__ == '__main__':
    csv_path = r'C:\Users\user1\Desktop\oshima\scene_sketch_recognition\results\result.csv'
    result_csv = ResultCsv(csv_path)
    model_id = result_csv.generate_id()
    result_csv.input_date(model_id)
    result_csv.input_name(model_id, 'model1')
    print(result_csv.pd_obj)
    result_csv.pd_obj.to_csv(csv_path, index=False)
    create_save_dir_path(model_id, 'model1')