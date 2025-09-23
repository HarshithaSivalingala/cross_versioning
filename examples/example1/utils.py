import numpy as np

def process_data(data):
    # Using deprecated numpy functions
    mean_val = np.asscalar(np.mean(data))
    data_float = data.astype(np.float)
    return data_float, mean_val