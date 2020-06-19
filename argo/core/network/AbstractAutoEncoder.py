from abc import abstractmethod

from .AbstractGenerativeModel import AbstractGenerativeModel
import tensorflow as tf
from ..utils.argo_utils import tf_add_gaussian_noise_and_clip, tf_clip, tf_rescale, tf_add_noise_to_discrete


import pdb

class AbstractAutoEncoder(AbstractGenerativeModel):
    default_params= {
        **AbstractGenerativeModel.default_params,
        "denoising" : 0, # enables denoising
        }
    
    def create_id(self):
        _id = '-d' + str(self._opts["denoising"]) #+\
          #'-re' + str(opts["rescale"])
          
        super_id = super().create_id()
        _id += super_id
        return _id
    
    def __init__(self, opts, dirName, check_ops=False, gpu=-1, seed=0):
        super().__init__(opts, dirName, check_ops, gpu, seed)
        self.denoising = opts["denoising"]
        
        # dictionaries with train, validation and test nodes
        self.x = None
        self.x_target = None


    def create_input_nodes(self, dataset):
        """
        creates input nodes for an autoencoder from the dataset

        Sets:
            x, x_target
        """
        
        datasets_nodes, handle, ds_initializers, ds_handles = self.create_datasets_with_handles(dataset)

        # optionally set y
        #if (not perturbed_dataset and len(datasets_nodes)==2) or (perturbed_dataset and len(datasets_nodes)==3):
        if len(datasets_nodes)==2:
            self.y = tf.identity(datasets_nodes[1], name="y")
            self.y_one_hot = tf.identity(tf.one_hot(self.y, dataset.n_labels), name="y1h")
            #self.y_one_hot = tf.identity(tf.cast(self.y, tf.float32), name="y1h")
        else:
            self.y = None
            self.y_one_hot = None

        # (Luigi) see _unpack_data_nodes(self, datasets_nodes) in AbstractGenerativeModel
        # where the order raw_x=raw_x, x_data=pert_x, and x_data_target=aug_x
        raw_x, x_data, x_data_target = self._unpack_data_nodes(datasets_nodes) #, perturbed_dataset)

        self.augment_bool = tf.placeholder_with_default(True, shape=())
        x_data, x_data_target = tf.cond(self.augment_bool,
                                        lambda: self._augment_data_nodes(x_data, x_data_target, self.denoising),
                                        lambda: (x_data, x_data_target)
                                        )
        self.raw_x = raw_x
        self.x = x_data
        self.x_target = x_data_target

    # this logic is very simple to follow and it seems good AFAIK, please talk to me if you need to modify it.. (Riccardo)
    def _augment_data_nodes(self, source, target, denoising=False):
        # AUGMENT SOURCE IF NEEDED
        if self.stochastic:
            if self.binary:
                source_before = source
                source = tf_add_noise_to_discrete(source, self.stochastic_noise_param)
                noise_data = source - source_before
            else:
                # TODO here I suppose the input are in -1.,1.
                # clip_after_noise is False if stochastic==2, see TFDeepLearningModel.py
                source, noise_data = tf_add_gaussian_noise_and_clip(source,
                                                                    std = self.stochastic_noise_param,
                                                                    clip_bool=self._clip_after_noise)

        # AUGMENT TARGET IF NEEDED
        if not denoising and self.stochastic:
            # use the same noise for continuous datasets
            target = tf_clip(target + noise_data)

        #TODO never rescale
        # #RESCALE BOTH
        # if not self.rescale==0.0:
        #     # TODO add a check that the domain is in [-1,1]
        #     source = tf_rescale(source, self.rescale)
        #     target = tf_rescale(target, self.rescale)

        return source, target

    '''
    def _unpack_data_nodes(self, datasets_nodes): #, is_perturbed_dataset):
        # what I will do next, is to move from
        #     dataset_x, perturbed_dataset_x
        # which are obtained from the dataset, to
        #     source_x, target_x
        # based on the value of perturbed_dataset

        source = None
        target = None

        
        # SOURCE NODE CREATE
        #if is_perturbed_dataset:
        #    dataset_x = datasets_nodes[0]
        #    perturbed_dataset_x = datasets_nodes[1]
        #    source = perturbed_dataset_x
        #else:
        #    dataset_x = datasets_nodes[0]
        #    source = dataset_x
        #
        #target = dataset_x

        if isinstance(datasets_nodes[0], tuple):
            target = datasets_nodes[0][0]
            source = datasets_nodes[0][1]
        else:
            target = datasets_nodes[0]
            source = datasets_nodes[0]
            
        return source, target
    '''

    @abstractmethod
    def encode(self, X, sess = None):
        pass

    @abstractmethod
    def decode(self, Z, sess = None):
        pass

    ''' # we don't have an easy wat for an autoencoder to generate samples
    @abstractmethod
    def generate(self):
        pass
    '''
    
    @abstractmethod
    def reconstruct(self, X):
        pass
