import numpy as np

__all__ = ["data_utils", "to_scalar"]


def to_scalar(value):
    array = np.asarray(value)
    return np.asscalar(array)
