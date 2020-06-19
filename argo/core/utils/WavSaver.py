import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

from scipy.io import wavfile

class WavSaver():

    def __init__(self, dirName, sample_rate):
        self._dirName = dirName
        self._sample_rate = sample_rate

    def save_wav(self, wav_data, fileName):
        wavfile.write(self._dirName + '/' + fileName + '.wav', self._sample_rate, wav_data)
