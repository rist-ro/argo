from abc import ABCMeta, abstractmethod

from .Model import Model

class OptimizationModel(Model):
    __metaclass__ = ABCMeta

    def __init__(self, seed=0):
        Model.__init__(self, seed)
        
    @abstractmethod
    def optimize(self):
        pass
