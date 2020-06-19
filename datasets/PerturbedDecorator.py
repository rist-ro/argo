#from core.TestAdvExamplesLauncher import TestAdvExamplesLauncher
#from datasets.Dataset import Dataset
#from datasets.MNIST import MNIST

import numpy as np

import pdb

class PerturbedDecorator(object):

    def __init__(self, perturbed_id):
        self._perturbed_id = perturbed_id
        self._train_set_x_fileName = None
        self._validation_set_x_fileName = None
        self._test_set_x_fileName = None
        
    @property
    def train_set_x_fileName(self):
        return self._train_set_x_fileName

    @train_set_x_fileName.setter
    def train_set_x_fileName(self, train_set_x_fileName):
        self._train_set_x_fileName = train_set_x_fileName

    @property
    def validation_set_x_fileName(self):
        return self._validation_set_x_fileName

    @validation_set_x_fileName.setter
    def validation_set_x_fileName(self, validation_set_x_fileName):
        self._validation_set_x_fileName = validation_set_x_fileName
        
    @property
    def test_set_x_fileName(self):
        return self._test_set_x_fileName

    @test_set_x_fileName.setter
    def test_set_x_fileName(self, test_set_x_fileName):
        self._test_set_x_fileName = test_set_x_fileName

    def __call__(self, cls):
        
        class PerturbedDataset(cls):

            perturbed_cls = cls

            # every time this class is defined as a new class
            # thus being static is not an issue
            perturbed_train_set_x_fileName = self._train_set_x_fileName
            perturbed_validation_set_x_fileName = self._validation_set_x_fileName
            perturbed_test_set_x_fileName = self._test_set_x_fileName
            perturbed_id = self._perturbed_id

            _perturbed_train_set_x = None
            _perturbed_validation_set_x = None
            _perturbed_test_set_x = None

            @property
            def perturbed_train_set_x(self):
                if self._perturbed_train_set_x is None and self.perturbed_train_set_x_fileName:
                    perturbed_train_set_x = np.load(self.perturbed_train_set_x_fileName)
                    self._perturbed_train_set_x = self.preprocess_x_y(perturbed_train_set_x, None)[0] # return only x, ignore y
                        
                return self._perturbed_train_set_x

            @property
            def perturbed_validation_set_x(self):
                if self._perturbed_validation_set_x is None and self.perturbed_validation_set_x_fileName:
                    perturbed_validation_set_x = np.load(self.perturbed_validation_set_x_fileName)
                    self._perturbed_validation_set_x = self.preprocess_x_y(perturbed_validation_set_x, None)[0] # return only x, ignore y
                
                return self._perturbed_validation_set_x
            
            @property
            def perturbed_test_set_x(self):
                if self._perturbed_test_set_x is None and self.perturbed_test_set_x_fileName:
                    perturbed_test_set_x = np.load(self.perturbed_test_set_x_fileName)
                    self._perturbed_test_set_x = self.preprocess_x_y(perturbed_test_set_x, None)[0] # return only x, ignore y
                
                return self._perturbed_test_set_x

            @property
            def id(self):
                return self._params["dataName"] + "-" + self.perturbed_id

        return PerturbedDataset

'''
launcher = TestAdvExamplesLauncher()
dataset_conf, algorith_parameters, algorithm_config = launcher.process_conf_file("/data2/adversarial/VAE-MNISTcontinuous/VAE_MNIST-c-st0_st1-stp0.1-bf0-d0-lr0.0001-c0-s3-en0-bs100-lvs17-nn200-200-reg0-actrelu-wixavier_init-bi0.1-k0-e0-eobs0-obs0-r0.conf")
r = AdversarialDecorator("advex_jan2018/jan16/","pgd_adv_test.npy")(MNIST)(dataset_conf)

print(r.test_set_x.shape)
#pdb.set_trace()
'''
