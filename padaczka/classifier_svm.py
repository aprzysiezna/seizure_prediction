from sklearn import svm
from padaczka.common import RegularizedClassifier


class Classifier(RegularizedClassifier):
    def __init__(self, C=10, kernel='rbf', gamma=0.01, coef0=0.0):
        self.clf = svm.SVC(C=C, kernel=kernel, gamma=gamma,
                           coef0=coef0, shrinking=True, probability=True,
                           tol=1e-5, cache_size=2000,
                           class_weight='balanced',
                           max_iter=-1)
        RegularizedClassifier.__init__(self)
