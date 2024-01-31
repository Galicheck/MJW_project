"""Using PCA to reduce dimensions of datasets"""

import pandas as pd
import config as cfg
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


k_red = 8
k_white = 8


def categorize(data):
    """Converting wine quality into ranges [0,4], [5,6], [7,8], [9,10]"""
    c_data = ["" for _ in range(len(data))]
    for i in range(len(data)):
        if data[i] <= 4:
            c_data[i] = 'Bad'
        elif data[i] <= 6:
            c_data[i] = 'Good'
        elif data[i] <= 8:
            c_data[i] = 'Great'
        else:
            c_data[i] = "Excellent"
    return c_data


def main():
    print('Preprocessing dataset...')
    red = pd.read_csv(cfg.General.red)
    white = pd.read_csv(cfg.General.white)
    red_x = red.loc[:, red.columns != 'quality']
    red_y = red['quality']
    white_x = white.loc[:, white.columns != 'quality']
    white_y = white['quality']

    scaler = StandardScaler().fit(red_x)
    red_x = scaler.transform(red_x)
    white_x = scaler.transform(white_x)

    pca_red = PCA(n_components=k_red).fit(red_x)
    pca_white = PCA(n_components=k_white).fit(white_x)
    red_x_pca = pca_red.transform(red_x)
    white_x_pca = pca_white.transform(white_x)

    df_red_pca_x = pd.DataFrame(data=red_x_pca,
                                columns=[f'PC_{i + 1}' for i in range(red_x_pca.shape[1])])
    df_red_pca_y = pd.DataFrame({'Quality': categorize(list(red_y))})
    df_white_pca_x = pd.DataFrame(data=white_x_pca,
                                  columns=[f'PC_{i + 1}' for i in range(white_x_pca.shape[1])])
    df_white_pca_y = pd.DataFrame({'Quality': categorize(list(white_y))})

    df_red_pca_x.to_csv(cfg.Preprocessing.red_x)
    df_red_pca_y.to_csv(cfg.Preprocessing.red_y)
    df_white_pca_x.to_csv(cfg.Preprocessing.white_x)
    df_white_pca_y.to_csv(cfg.Preprocessing.white_y)


if __name__ == '__main__':
    main()
