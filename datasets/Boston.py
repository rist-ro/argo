import numpy as np
import os

import pdb

from sklearn.datasets import load_boston

def load():
    boston = load_boston()    
    
    ds={}

    n_points = boston.data.shape[0]
    perm = np.random.permutation(n_points)

    percentage_train = 0.7
    
    ds["n_samples_train"] = int(percentage_train*n_points) 
    ds["n_samples_test"] = n_points - int(percentage_train*n_points)
    ds["train_set_x"] = boston.data[:int(percentage_train*n_points)]
    ds["train_set_y"] = boston.target[:int(percentage_train*n_points)]
    ds["test_set_x"] = boston.data[int(percentage_train*n_points):]
    ds["test_set_y"] = boston.target[int(percentage_train*n_points):]
    
    ds["input_size"] = 13
    ds["output_size"] = 1

    ds["binary"] = 0

    return ds
    
########################################################################
