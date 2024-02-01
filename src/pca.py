"""Preprocessing of the dataset"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from dvc.api import params_show
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import config as cfg

params = params_show()['pca']
k_red = params['k_red']
k_white = params['k_white']


def preprocessing():
    """Clearing data and parting for red and white wine"""
    print('Preprocessing dataset...')
    df = pd.read_csv(cfg.General.raw_data)
    df = df.dropna()
    red = df[df['type'] == 'red']
    red = red.drop(['type'], axis=1)
    white = df[df['type'] == 'white']
    white = white.drop(['type'], axis=1)
    return red, white


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


def pca(data_red, data_white):
    print('Initializing PCA...')

    red = data_red
    """ Division into sets X and Y."""
    red_x = red.loc[:, red.columns != 'quality']
    red_y = red['quality']
    """Standarizing data before and after PCA."""
    scaler = StandardScaler().fit(red_x)
    red_x = scaler.transform(red_x)
    pca_red = PCA(n_components=k_red).fit(red_x)
    red_x_pca = pca_red.transform(red_x)

    df_red_pca_x = pd.DataFrame(data=red_x_pca,
                                columns=[f'PC_{i + 1}' for i in range(red_x_pca.shape[1])])
    df_red_pca_y = pd.DataFrame({'Quality': categorize(list(red_y))})



    white = data_white
    white_x = white.loc[:, white.columns != 'quality']
    white_y = white['quality']

    scaler = StandardScaler().fit(white_x)
    white_x = scaler.transform(white_x)
    pca_white = PCA(n_components=k_white).fit(white_x)
    white_x_pca = pca_white.transform(white_x)

    df_white_pca_x = pd.DataFrame(data=white_x_pca,
                                  columns=[f'PC_{i + 1}' for i in range(white_x_pca.shape[1])])
    df_white_pca_y = pd.DataFrame({'Quality': categorize(list(white_y))})

    df_red_pca_x.to_csv(cfg.PCA.red_x)
    df_red_pca_y.to_csv(cfg.PCA.red_y)
    df_white_pca_x.to_csv(cfg.PCA.white_x)
    df_white_pca_y.to_csv(cfg.PCA.white_y)

    return df_red_pca_x, df_red_pca_y, df_white_pca_x, df_white_pca_y

def main():
    red, white = preprocessing()
    pca(red, white)


if __name__ == '__main__':
    main()