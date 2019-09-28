import pandas as pd
import os
from os import path
import configparser
import re
sep = os.sep


class PreProcessing():
    def __init__(self):
        # Read config
        current_dir = path.dirname(path.abspath(__file__))
        config_file = current_dir + sep + 'config-preprocessing.ini'
        abs_path_config = os.path.abspath(config_file)

        self._config = configparser.ConfigParser()
        self._config.optionxform = str  # differ with large and small character
        self._config.read(abs_path_config, encoding="utf-8")

        # check config
        chk_sections = ('INPUT', 'OUTPUT')
        for section in chk_sections:
            if section not in self._config.sections():
                raise_msg = 'invalid ini file has no section {}'.format(section)
                raise ValueError(raise_msg)

        check_input_key = ['train', 'test', 'sample_submit']
        check_output_key = ['output_path']

        self._INPUT = dict(self._config.items('INPUT'))
        self._OUTPUT = dict(self._config.items('OUTPUT'))

        for key in check_input_key:
            if key not in self._INPUT:
                raise_msg = 'invalid INPUT section has no key {}'.format(key)
                raise ValueError(raise_msg)

        for key in check_output_key:
            if key not in self._OUTPUT:
                raise_msg = 'invalid OUTPUT section has no key {}'.format(key)
                raise ValueError(raise_msg)

        self._train_path = self._INPUT['train']
        self._test_path = self._INPUT['test']
        self._sample_submit_path = self._INPUT['sample_submit']

        self._output_path = self._OUTPUT['output_path']

        self._train = pd.read_csv(self._train_path)
        self._test = pd.read_csv(self._test_path)
        self._sample_submit = pd.read_csv(self._sample_submit_path)

    def process(self, data):
        location = data['所在地']
        access = data['アクセス']
        layout = data['間取り']
        build_time = data['築年数']
        direction = data['方角']
        area = data['面積']
        floor = data['所在階']
        bath_toilet = data['バス・トイレ']
        kitchen = data['キッチン']
        communication = data['放送・通信']
        equipment = data['室内設備']
        parking = data['駐車場']
        environment = data['周辺環境']
        structure = data['建物構造']
        contract = data['契約期間']

        # process
        data['アクセス'] = self.process_access(data['アクセス'])
        data['間取り'] = self.process_layout(data['間取り'])
        data['築年数'] = self.process_build_time(data['築年数'])
        data['面積'] = self.process_area(data['面積'])
        data['建物構造'] = self.process_structure(data['建物構造'])
        data['バス・トイレ'] = self.process_bath_toilet(data['バス・トイレ'])
        data['方角'] = self.process_direction(data['方角'])

        drop_columns = ['id', '所在地', '契約期間', 'キッチン', '放送・通信', '周辺環境', '所在階', '室内設備', '駐車場']
        data.drop(drop_columns, axis=1, inplace=True)
        data = data.astype("float64")
        return data

    def get_train_data(self):
        dataset = self.process(self._train)
        label = dataset['賃料']
        train = dataset.drop('賃料', axis=1)
        train = self.norm(train)
        return train, label

    def get_test_data(self):
        test = self.process(self._test)
        test = self.norm(test)
        return test

    @staticmethod
    def norm(dataset):
        stats = dataset.describe()
        stats = stats.transpose()
        return (dataset - stats["mean"]) / stats["std"]

    @staticmethod
    def process_area(area):
        area = area.str.replace("m2", "")
        area.astype("float32")
        return area

    @staticmethod
    def process_build_time(build_time):
        def build_time_f_str(x):
            str_arr = x.split(".")
            year = int(str_arr[0])
            month = 0
            if len(str_arr) > 1:
                month = int(str_arr[1])
            return year * 12 + month
        build_time = build_time.str.replace("年", ".")
        build_time = build_time.str.replace("ヶ月", "")
        build_time = build_time.str.replace("新築", "0")
        # 築（月）数　　大きければ大きいほど悪い
        build_time = build_time.map(build_time_f_str)
        return build_time

    @staticmethod
    def process_access(access):
        def take_min_number_in_text(text):
            pattern = r'([0-9]*)'
            lists = re.findall(pattern, text)
            number_list = []
            for item in lists:
                if item.isdigit():
                    number_list.append(item)
            return int(min(number_list))
        access = access.map(take_min_number_in_text)
        return access

    @staticmethod
    def process_layout(layout):
        layout_level = {'1R': 1, '1K': 2, '1K+S(納戸)': 3, '1DK': 4, '1DK+S(納戸)': 5,
                        '1LK+S(納戸)': 6, '1LDK': 7, '1LDK+S(納戸)': 8,
                        '2R':6, '1LK': 5,
                        '2K': 9, '2K+S(納戸)': 10, '2DK': 11, '2DK+S(納戸)': 12,
                        '2LDK': 13, '2LDK+S(納戸)': 14,
                        '3K': 15, '3K+S(納戸)': 16, '3DK': 17, '3DK+S(納戸)': 18,
                        '3LDK': 19, '3LDK+S(納戸)': 20,
                        '4K': 21, '4DK': 22, '4LDK': 23, '4LDK+S(納戸)': 24, '4DK+S(納戸)': 24,
                        '5K': 25, '5DK': 26, '5DK+S(納戸)': 27, '5LDK': 28,
                        '5LDK+S(納戸)': 29, '6LDK': 30,
                        '6K':31,'6DK':31, '6LDK+S(納戸)':31,'8LDK':32, '11R':1}

        def map_layout_level(key):
            if key is not None:
                return layout_level[key]
            else:
                return None

        layout = layout.map(map_layout_level)
        return layout

    @staticmethod
    def process_structure(structure):
        structure_level = {
            'RC（鉄筋コンクリート）': 9, '鉄骨造': 8, '木造': 7, 'SRC（鉄骨鉄筋コンクリート）': 6, '軽量鉄骨': 5,
            'ALC（軽量気泡コンクリート）': 4, 'その他': 0, 'PC（プレキャスト・コンクリート（鉄筋コンクリート））': 3,
            'HPC（プレキャスト・コンクリート（重量鉄骨））': 2, 'ブロック': 1,
            '鉄筋ブロック': 8,
        }

        def map_structure_level(key):
            if key is not None:
                return structure_level[key]
            else:
                return None

        structure = structure.map(map_structure_level)
        return  structure

    @staticmethod
    def process_bath_toilet(bath_toilet):
        def function_str(text):
            point_map = {"専用バス": 1,
                         "専用トイレ": 1,
                         "バス・トイレ別": 1,
                         "シャワー": 1,
                         "追焚機能": 1,
                         "浴室乾燥機": 1,
                         "温水洗浄便座": 1,
                         "洗面台独立": 1,
                         "脱衣所": 1,
                         "共同トイレ": -1,
                         }
            total_point = 0
            if type(text) is str:
                arr = text.split("／\t")
                for item in arr:
                    if item in point_map.keys():
                        total_point += point_map[item]
            return total_point
        return bath_toilet.map(function_str)

    @staticmethod
    def process_direction(direction):
        def function_str_direction(text):
            direction_level = {
                '南東': 3, '南': 3, '東': 2, '北西': 1, '西': 1, '北': 1, '南西': 1, '北東': 2}
            if text in direction_level.keys():
                return direction_level[text]
            else:
                return 0

        return direction.map(function_str_direction)

if __name__ == '__main__':
    instance = PreProcessing()
    train, label = instance.get_train_data()
    print(label)
    print(train)