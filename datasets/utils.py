import numpy as np

def normalize(tensor, min_orig, max_orig, min_out=-1., max_out=1.):
    delta = max_out - min_out
    return delta * (tensor - min_orig) / (max_orig - min_orig) + min_out

def min_max_data_np(arrays):
    all_max = []
    all_min = []

    for arr in arrays:
        all_min.append(np.min(arr))
        all_max.append(np.max(arr))

    data_min = np.min(all_min)
    data_max = np.max(all_max)

    return data_min, data_max

