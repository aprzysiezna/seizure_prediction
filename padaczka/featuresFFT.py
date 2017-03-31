import numpy as np
from scipy.signal import resample
from scipy.fftpack import rfft
from padaczka.common import cmatrix_with_evals


def compute_features(data):
    """ Return  features for data sequence. """
    if np.allclose(data, 0):
        # nothing to do...
        return np.empty(0)

    power = np.absolute(rfft(data, axis=0))[50:2500]
    resampled = resample(power, num=18, axis=0)
    resampled[np.less_equal(resampled, 0)] = 1e-6  
    logfreq = np.log10(resampled)
    features = [logfreq.ravel(),
                cmatrix_with_evals(logfreq)]
    return np.concatenate(features, axis=0)
