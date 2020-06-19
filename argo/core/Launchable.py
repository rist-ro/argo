from abc import ABC, abstractmethod, abstractclassmethod

from .utils.argo_utils import update_conf_with_defaults

import numpy as np
# import ipdb as pdb

class Launchable(ABC):

    default_params = {}
    launchable_name = "Launchable"

    def __init__(self, opts, dirName, seed=0):

        self._opts_original = opts
        self._opts = update_conf_with_defaults(opts, self.default_params)

        self._id = self.create_id() + self.run_id(opts["run"])

        np.random.seed(seed)
        self._seed = seed

        self.dirName = dirName + "/" + self._id


    @abstractmethod
    def create_id(self):
        return ""
    
    def run_id(self, run):
        _id = '-r' + str(run)
        return _id
    
    # @abstractmethod
    # def run(self):
    #     pass
    #

    @property
    def id(self):
        """
        Returns th id of the Launchable
        :return: the id
        """
        return self._id

    def init(self):
        pass

    # #TODO-ARGO2 no need for this at this level... every child can log if and how he wants
    # def register_logger(self, logger, name):
    #     """Register a logger.
    #
    #     Args:
    #         logger (argo.core.Logger): A logger to register.
    #         name (string): the key under which to register the logger in the logger list self._loggers.
    #
    #     """
    #     if name in self._loggers:
    #         raise ValueError("Cannot register logger `%s`. Logger already exist."%name)
    #
    #     self._loggers[name] = logger


    def release(self):
        """Release the resources of all the loggers."""
        pass
    
        #for log in self._loggers:
        #    self._loggers[log].release()

    # @staticmethod
    # def get_default_params():
    #     """ returns the dictionary with default values"""
    #     return Launchable.default_params

    @classmethod
    def add_default_parameters(cls, params):
        """
        Preprocessing of the parameters from config and argo.

        In practice it updates the default values with the ones produced by
        argo from the config file.

        Parameters
        ----------
        params : dict
            parameters extreacted from the config file

        Returns
        -------
        full_params : dict
            the model_params dictionary where missing values are replace by default
            ones.

        """

        full_params = cls.default_params.copy()
        # check the dimension
        #if len(full_params) + 2 < len(model_params): # run are introduced dynamically
        #    print("Warning: argo passed more options then those in default")

        # update the model_params dictionary
        full_params.update(params)

        # # unpack full_params
        # for k in full_params.keys():
        #     if type(full_params[k]) == list:
        #         if len(full_params[k]) != 1:
        #             raise Exception(
        #                 'The option {} is not a single valued list'.format(k))
        #         else:
        #             full_params[k] = full_params[k][0]

        return full_params

    @property
    def seed(self):
        return self._seed
