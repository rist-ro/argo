import tensorflow as tf
import numpy as np
from argo.core.TFDeepLearningModel import load_model, load_network, load_model_without_session

import os

d_y = np.load("iStet-AE-train-y.npy", allow_pickle=True)
d_z = np.load("iStet-AE-train-z.npy", allow_pickle=True)
d_v_y = np.load("iStet-AE-validation-y.npy", allow_pickle=True)
d_v_z = np.load("iStet-AE-validation-z.npy", allow_pickle=True)

print(d_y.shape, d_z.shape, d_v_y.shape, d_v_z.shape)

d_y = np.load("DigiScope-AE-train-y.npy", allow_pickle=True)
d_z = np.load("DigiScope-AE-train-z.npy", allow_pickle=True)
d_v_y = np.load("DigiScope-AE-validation-y.npy", allow_pickle=True)
d_v_z = np.load("DigiScope-AE-validation-z.npy", allow_pickle=True)

print(d_y.shape, d_z.shape, d_v_y.shape, d_v_z.shape)
