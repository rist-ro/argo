"""
Module for managing image datasets.

Introduce the classe ImageDataset that provide utils for deaing with picture
"""

from abc import ABCMeta, abstractmethod
from .Dataset import Dataset
import tensorflow as tf

class ImageDataset(Dataset):
    
    def __init__(self, params):
        super().__init__(params)


    @property
    def image_shape(self):
        return self.x_shape # 1 is the number of channels