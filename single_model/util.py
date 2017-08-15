
import sys
import os
import time
import logging
import ConfigParser
import operator
import numpy as np
import pandas as pd


def prepare_data():
    pass


def run_model(config_name, model_dict, data_path):
    config = ConfigParser.ConfigParser()
    try:
        config.read("models.config")
        valid_mode_on = config.getboolean(config_name, "valid_mode_on")
        if valid_mode_on:
            train_path = os.path.join(data_path, "train-tr.csv")
            test_path = os.path.join(data_path, "train-va.csv")
        else:
            train_path = os.path.join(data_path, "train.csv")
            test_path = os.path.join(data_path, "test.csv")
        model_list = map(lambda x: model_dict[x.strip()], config.get(config_name, "model_list").split(","))
        output_path = os.path.join(config.get(config_name, "output_path"), config_name)

    except Exception as e:
        logging.error("Could not load configuration file from models.config")
        logging.error(str(e))

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    logging.info("Running config: [%s]" % config_name)
    logging.info('Loading data')
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    logging.info('Preparing train data')

