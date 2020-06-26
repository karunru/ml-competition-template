import numpy as np

# https://www.guruguru.ml/competitions/10/discussions/c26ab5b4-2646-49ca-b040-e014cb68a1ef/


def median_absolute_deviation(x):
    return np.median(np.abs(x - np.median(x)))


def mean_variance(x):
    return np.std(x) / np.mean(x)


def hl_ratio(x):
    return np.sum(x >= np.mean(x)) / np.sum(x < np.mean(x))