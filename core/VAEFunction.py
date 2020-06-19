import math

from abc import ABCMeta, abstractmethod

#import tensorflow as tf

class VAEFunction:
    __metaclass__ = ABCMeta

    def __init__(self):
        self._cost_logger_class = None

    @abstractmethod
    def compute(self):
        pass
    
