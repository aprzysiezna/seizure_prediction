import numpy as np
from sklearn import preprocessing  # TODO check that
from sklearn import linear_model


def cmatrix(data):
    """ Returns upper triangle of correlation matrix
    and its sorted eigenvalues. """
    cm = np.corrcoef(preprocessing.scale(data, axis=1), rowvar=0)
    n = cm.shape[0]
    return cm[np.triu_indices(n, 1)]


def cmatrix_with_evals(data):
    """ Returns upper triangle of correlation matrix
    and its sorted eigenvalues. """
    cm = np.corrcoef(preprocessing.scale(data, axis=1), rowvar=0)
    n = cm.shape[0]
    return np.concatenate([cm[np.triu_indices(n, 1)],
                           np.sort(np.fabs(np.linalg.eigvalsh(cm)))],
                          axis=0)


def data_sequences(data, dt=50):
    freq = 400.0
    n = data.shape[0]
    breaks = np.arange(0, n, dt * freq).astype('int')
    return (data[start:stop] for start, stop in zip(breaks[:-1], breaks[1:]))


class RegularizedClassifier(object):
    """ Boilerplate for logistic regression calibration of model.
    Just derive this class and set self.clf of parent to actuall
    classifier. """
    def __init__(self):
        self.cal = linear_model.LogisticRegression(random_state=0)

    def fit(self, data, target):
        self.clf.fit(data, target)

        # calibration
        est = self.clf.predict_proba(data)
        self.cal.fit(est, target)

    def predict_proba(self, data):
        return self.cal.predict_proba(self.clf.predict_proba(data))