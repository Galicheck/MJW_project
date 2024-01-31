"""Preprocessing of the dataset"""

import pandas as pd
import config as cfg


def main():
    print('Preprocessing dataset...')
    df = pd.read_csv(cfg.General.raw_data)
    df = df.dropna()
    red = df[df['type'] == 'red']
    red = red.drop(['type'], axis=1)
    white = df[df['type'] == 'white']
    white = white.drop(['type'], axis=1)
    red.to_csv(cfg.General.red)
    white.to_csv(cfg.General.white)


if __name__ == '__main__':
    main()
