from abc import abstractmethod

#import numpy as np

from .Launchable import Launchable

#TODO-ARGO2 this class does not seem very useful
class DeepLearningModel(Launchable):
    
    def __init__(self, opts, dirName, seed=0):
        super().__init__(opts, dirName, seed)
        
    @abstractmethod
    def train(self):
        pass

    # @abstractmethod
    # def test(self):
    #     pass
    
    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def restore(self):
        pass
