from dvc.api import params_show
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import ConfusionMatrixDisplay
import config as cfg
from sklearn.model_selection import train_test_split
# import pandas was showing a warning
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

# Fetching parameters for the SGD classifier
params = params_show()['sgd_classifier']

# Assigning parameter values to variables
loss = params['loss']  # Loss function type
alpha = params['alpha']  # Regularization coefficient
penalty = params['penalty']  # Regularization type
max_iter = params['max_iter']  # Maximum number of iterations
test_size = params['test_size']  # Test set size


def SGD():
    """ Adapting SGD Classifier for red wine"""
    x = pd.read_csv(cfg.PCA.red_x)
    y = list(pd.read_csv(cfg.PCA.red_y)['Quality'])

    trainx, testx, trainy, testy = train_test_split(x, y, test_size=test_size)

    scaler = StandardScaler()
    scaler.fit(trainx)
    trainx = scaler.transform(trainx)
    testx = scaler.transform(testx)

    clf = SGDClassifier(loss=loss, alpha=alpha, penalty=penalty, max_iter=max_iter)
    clf.fit(trainx, trainy)

    y_pred = clf.predict(testx)
    score1 = round(accuracy_score(testy, y_pred)*100,2)
    print(str(score1)+"%")

    matrix = cm(testy, y_pred, labels=['Bad', 'Good', 'Great', 'Excellent'])
    ConfusionMatrixDisplay(matrix, display_labels=['Bad', 'Good', 'Great', 'Excellent']).plot()
    ax = plt.gca()
    ax.set_title('Confusion Matrix for red wine ')
    plt.savefig('plots/confusion_red.png')
    plt.show()

    """ Adapting SGD Classifier for white wine"""
    x = pd.read_csv(cfg.PCA.white_x)
    y = list(pd.read_csv(cfg.PCA.white_y)['Quality'])

    trainx, testx, trainy, testy = train_test_split(x, y, test_size = test_size)

    scaler = StandardScaler()
    scaler.fit(trainx)
    trainx = scaler.transform(trainx)
    testx = scaler.transform(testx)

    clf = SGDClassifier(loss=loss, alpha=alpha, penalty=penalty, max_iter=max_iter)
    clf.fit(trainx, trainy)

    y_pred = clf.predict(testx)
    score2 = round(accuracy_score(testy, y_pred)*100,2)
    print(str(score2)+"%")

    matrix = cm(testy, y_pred, labels=['Bad', 'Good', 'Great', 'Excellent'])
    ConfusionMatrixDisplay(matrix, display_labels=['Bad', 'Good', 'Great', 'Excellent']).plot()
    ax = plt.gca()
    ax.set_title('Confusion Matrix for red wine ')
    plt.savefig('plots/confusion_white.png')
    plt.show()

    return score1, score2


if __name__ == '__main__':
    SGD()
