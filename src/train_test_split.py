"""Splits the preprocessed datasets into test/train subsets"""

import pandas as pd
import config as cfg
from sklearn.model_selection import train_test_split


test_size = 0.2


def main():
    print('Splitting dataset into train/test subsets...')
    red_x = pd.read_csv(cfg.Preprocessing.red_x)
    red_y = list(pd.read_csv(cfg.Preprocessing.red_y)['Quality'])
    r_train_x, r_test_x, r_train_y, r_test_y = train_test_split(red_x, red_y, test_size=test_size)
    r_train_x.to_csv(cfg.TrainTestSplit.red_trainX)
    r_test_x.to_csv(cfg.TrainTestSplit.red_testX)
    r_train_y = pd.DataFrame(r_train_y)
    r_train_y.to_csv(cfg.TrainTestSplit.red_trainY, index=False)
    r_test_y = pd.DataFrame(r_test_y)
    r_test_y.to_csv(cfg.TrainTestSplit.red_testY, index=False)

    white_x = pd.read_csv(cfg.Preprocessing.white_x)
    white_y = list(pd.read_csv(cfg.Preprocessing.white_y)['Quality'])
    w_train_x, w_test_x, w_train_y, w_test_y = train_test_split(white_x, white_y, test_size=test_size)
    w_train_x.to_csv(cfg.TrainTestSplit.white_trainX)
    w_test_x.to_csv(cfg.TrainTestSplit.white_testX)
    w_train_y = pd.DataFrame(w_train_y)
    w_train_y.to_csv(cfg.TrainTestSplit.white_trainY, index=False)
    w_test_y = pd.DataFrame(w_test_y)
    w_test_y.to_csv(cfg.TrainTestSplit.white_testY, index=False)


if __name__ == '__main__':
    main()
