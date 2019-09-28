from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
import time
import json
from keras import regularizers
from keras import backend as K
import numpy as np
from sklearn.model_selection import StratifiedKFold
K.tensorflow_backend._get_available_gpus()

import pandas as pd
import os
from os import path
import configparser
import re
import sys
# for training
from sklearn.model_selection import train_test_split
import math
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, roc_curve
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
sys.path.insert(0,'./../')
from preprocessing.preprocessing import PreProcessing

sep = os.sep


# エポックが終わるごとにドットを一つ出力することで進捗を表示
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print("")
        print(".", end="")


class Engine():
    def __init__(self):
        self.preprocessing = PreProcessing()

        # grid search
        params = {"learning_rate": [0.1],
                  "max_depth": [5],
                  "subsample": [0.9],
                  "min_samples_leaf": [50],
                  "min_samples_split": [100],
                  "colsample_bytree": [0.5],
                  "n_estimators": [1000]
                  }

        mod = xgb.XGBRegressor()
        self.cv = GridSearchCV(mod, params, cv=3, scoring='r2', n_jobs=-1, verbose=2)
        self.train_data, self.train_labels = self.preprocessing.get_train_data()
        self.test_data = self.preprocessing.get_test_data()

    def xgboost_fit(self):
        # X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.3,
        #                                                     random_state=0)
        self.cv.fit(self.train_data, self.train_labels)

        # Return the optimal model after fitting the data
        return self.cv.best_estimator_

    def nn_train(self):
        # model
        layer_numbers = 256

        def root_mean_squared_error(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true)))

            # mean_squared_logarithmic_error, mean_absolute_percentage_error, mean_absolute_error
        loss = root_mean_squared_error

        def build_model(dataset):
            model = keras.Sequential([
                layers.Dense(128,  kernel_initializer='normal', input_shape=[len(self.train_data.keys())],
                             activation=tf.nn.relu),
                layers.Dense(layer_numbers, kernel_initializer='normal', activation=tf.nn.relu),
                layers.Dense(layer_numbers, kernel_initializer='normal', activation=tf.nn.relu),
                layers.Dense(layer_numbers, kernel_initializer='normal', activation=tf.nn.relu),
                layers.Dense(1, kernel_initializer='normal', activation='linear')
            ])
            # optimizer = tf.keras.optimizers.RMSprop(0.001)
            optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
            model.compile(
                loss = loss,
                optimizer=optimizer,
                metrics=["mean_squared_error", "mean_absolute_percentage_error"],
            )
            return model

        model = build_model(self.train_data)

        # if os.path.isfile("param_goto.hdf5"):
        #     model.load_weights('param_goto.hdf5')

        model.summary()
        # example trainning
        example_batch = self.train_data[:10]
        example_result = model.predict(example_batch)
        example_result

        return model

    def nn_train_fit(self):
        def plot_history(history):
            hist = pd.DataFrame(history.history)
            hist["epoch"] = history.epoch

            plt.figure()
            plt.xlabel("Epoch")
            plt.ylabel("loss")
            plt.plot(hist["epoch"], hist["loss"], label="Train Error")
            plt.plot(hist["epoch"], hist["val_loss"], label="Val Error")
            plt.legend()

            plt.figure()
            plt.xlabel("Epoch")
            plt.ylabel("Mean Abs Percentage Error")
            plt.plot(hist["epoch"], hist["mean_absolute_percentage_error"], label="Train Error")
            plt.plot(hist["epoch"], hist["val_mean_absolute_percentage_error"], label="Val Error")
            plt.legend()
            plt.show()

        # callback when NG
        EPOCHS = 100

        start_time = time.time()
        # The patience parameter is the amount of epochs to check for improvement
        early_stop = keras.callbacks.EarlyStopping(monitor='val_mean_absolute_percentage_error', patience=10)

        model = self.nn_train()

        history = model.fit(self.train_data, self.train_labels, batch_size=128, epochs=EPOCHS, shuffle=True,
                            validation_data=(self.train_data, self.train_labels), verbose=2,
                            callbacks=[early_stop, PrintDot()])
        end_time = time.time()
        execute_time = end_time - start_time
        print("execute time: {}".format(execute_time))
        plot_history(history)

        # save model
        json_string = model.to_json()
        with open('model_goto.json', 'w') as outfile:
            json.dump(json_string, outfile)

        # save weight
        model.save_weights('param_goto.hdf5')

        return model


if __name__ == '__main__':
    instance = Engine()



