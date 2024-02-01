import warnings
warnings.filterwarnings("ignore")
from dvc.api import params_show
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import ConfusionMatrixDisplay as CMD
import config as cfg
from sklearn.model_selection import train_test_split
import pandas as pd

params = params_show()['sgd_classifier']
loss = params['loss']
alpha = params['alpha']
penalty = params['penalty']
max_iter = params['max_iter']
test_size = params['test_size']

def SGD():
    """ Adapting SGD Classifier for red wine"""
    x = pd.read_csv(cfg.PCA.red_x)
    y = list(pd.read_csv(cfg.PCA.red_y)['Quality'])

    trainX, testX, trainY, testY = train_test_split(x, y, test_size = test_size)

    scaler = StandardScaler()
    scaler.fit(trainX)
    trainX = scaler.transform(trainX)
    testX = scaler.transform(testX)

    clf = SGDClassifier(loss=loss, alpha=alpha, penalty=penalty, max_iter=max_iter)
    clf.fit(trainX, trainY)

    y_pred = clf.predict(testX)
    score1 = round(accuracy_score(testY, y_pred)*100,2)
    print(str(score1)+"%")

    matrix = CM(testY, y_pred, labels=['Bad', 'Good', 'Great', 'Excellent'])
    CMD(matrix, display_labels=['Bad', 'Good', 'Great', 'Excellent']).plot()
    ax = plt.gca()
    ax.set_title('Confusion Matrix for red wine ')
    plt.savefig('plots/confusion_red.png')
    plt.show()



    """ Adapting SGD Classifier for white wine"""
    x = pd.read_csv(cfg.PCA.white_x)
    y = list(pd.read_csv(cfg.PCA.white_y)['Quality'])

    trainX, testX, trainY, testY = train_test_split(x, y, test_size = test_size)

    scaler = StandardScaler()
    scaler.fit(trainX)
    trainX = scaler.transform(trainX)
    testX = scaler.transform(testX)

    clf = SGDClassifier(loss=loss, alpha=alpha, penalty=penalty, max_iter=max_iter)
    clf.fit(trainX, trainY)

    y_pred = clf.predict(testX)
    score2 = round(accuracy_score(testY, y_pred)*100,2)
    print(str(score2)+"%")

    matrix = CM(testY, y_pred, labels=['Bad', 'Good', 'Great', 'Excellent'])
    CMD(matrix, display_labels=['Bad', 'Good', 'Great', 'Excellent']).plot()
    ax = plt.gca()
    ax.set_title('Confusion Matrix for red wine ')
    plt.savefig('plots/confusion_white.png')
    plt.show()



    return score1, score2

if __name__ == '__main__':
    SGD()

