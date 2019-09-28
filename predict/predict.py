import pandas as pd
import os
from os import path
import configparser
import re
import datetime
import sys
sys.path.insert(0,'./../')
from engine.engine import Engine
from preprocessing.preprocessing import PreProcessing
sep = os.sep


class Predict():
    def __init__(self):
        self.preprocess = PreProcessing()
        self.engine = Engine()

        # model
        model = self.engine.nn_train_fit()

        test_data = self.preprocess.get_test_data()
        prediction = model.predict(test_data)
        print(prediction)
        self.save_submit(prediction)

    def save_submit(self, prediction):
        submit = pd.read_csv("./../Data/sample_submit.csv", header=None)
        submit[1] = prediction
        now = datetime.datetime.now()
        now_str = '{}_{}_{}_{}_{}'.format(now.year, now.month, now.day, now.hour, now.minute)
        submit_folder = "./../Data/submit"
        if not os.path.exists(submit_folder):
            os.mkdir(submit_folder)
        submit_file = submit_folder + '/submit_{}.csv'.format(now_str)
        submit.to_csv(submit_file, header=None, index=None)


if __name__ == '__main__':
    instance = Predict()



