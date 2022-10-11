from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def normalizeUsingMinMaxScaler(X):
    return MinMaxScaler().fit_transform(X)

def normalizeUsingStandardization(X):
    return StandardScaler().fit_transform(X)

def chooseNormalizationMethod(method, X):
    switcher = {
        0: normalizeUsingStandardization(X),
        1: normalizeUsingMinMaxScaler(X)
    }

    return switcher.get(method)