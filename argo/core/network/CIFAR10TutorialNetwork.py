import tensorflow as tf
import sonnet as snt
import numpy as np

from operator import xor

import pdb

from abc import ABC, abstractmethod

from tensorflow_probability import distributions as tfd

from .AbstractModule import AbstractModule

from ..utils.argo_utils import NUMTOL
import types

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
#tf.app.flags.DEFINE_integer('batch_size', 128,
#                            """Number of images to process in a batch.""")
#tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
#                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

class CIFAR10TutorialNetwork(AbstractModule):
    
    def __init__(self, size=-1, shape=-1,
                 initializers = {}, regularizers = {},
                 name='CIFAR10TutorialNetwork'):

        super().__init__(name = name)

        '''
        assert xor(size==-1,shape==-1), "Either size or shape mut be specified, not both"
        
        if size!=-1:
            self._shape = [size]
        else:
            self._shape = shape
        '''

    #see https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py~
    def _build(self, inputs):
       
        """Build the CIFAR-10 model.
        Args:
        images: Images returned from distorted_inputs() or inputs().
        Returns:
        Logits.
        """
        
        # We instantiate all variables using tf.get_variable() instead of
        # tf.Variable() in order to share variables across multiple GPU training runs.
        # If we only ran this model on a single GPU, we could simplify this function
        # by replacing all instances of tf.get_variable() with tf.Variable().

        NUM_CLASSES = 10
        
        # conv1
        with tf.variable_scope('conv1') as scope:
            kernel = self._variable_with_weight_decay('weights',
                                                      shape=[5, 5, 3, 64],
                                                      stddev=5e-2,
                                                      wd=None)
            conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self._variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
            #_activation_summary(conv1)

        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')
        # norm1
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm1')

        # conv2
        with tf.variable_scope('conv2') as scope:
            kernel = self._variable_with_weight_decay('weights',
                                                      shape=[5, 5, 64, 64],
                                                      stddev=5e-2,
                                                      wd=None)
            conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self._variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)
            #_activation_summary(conv2)

        # norm2
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm2')
        # pool2
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # local3
        with tf.variable_scope('local3') as scope:
            # Move everything into depth so we can perform a single matrix multiply.

            #reshape = tf.reshape(pool2, [inputs.get_shape().as_list()[0], -1])
            l = int(np.prod(inputs.get_shape().as_list()[1:])/3*4)
            #pdb.set_trace()
            reshape = tf.reshape(pool2, [-1, l])
            dim = reshape.get_shape()[1].value
            weights = self._variable_with_weight_decay('weights', shape=[dim, 300],
                                                       stddev=0.04, wd=0.004)
            biases = self._variable_on_cpu('biases', [300], tf.constant_initializer(0.1))
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
            #_activation_summary(local3)

        # local4
        with tf.variable_scope('local4') as scope:
            weights = self._variable_with_weight_decay('weights', shape=[300, 150],
                                                       stddev=0.04, wd=0.004)
            biases = self._variable_on_cpu('biases', [150], tf.constant_initializer(0.1))
            local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
            #_activation_summary(local4)
            
        # linear layer(WX + b),
        # We don't apply softmax here because
        # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
        # and performs the softmax internally for efficiency.
        with tf.variable_scope('softmax_linear') as scope:
            weights = self._variable_with_weight_decay('weights', [150, NUM_CLASSES],
                                                       stddev=1/150.0, wd=None)
            biases = self._variable_on_cpu('biases', [NUM_CLASSES],
                                      tf.constant_initializer(0.0))
            softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
            #_activation_summary(softmax_linear)
    
        return softmax_linear

    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.
        Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
        Returns:
        Variable Tensor
        """
        
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = self._variable_on_cpu(name,
                                    shape,
                                    tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _variable_on_cpu(self, name, shape, initializer):
        """Helper to create a Variable stored on CPU memory.
        Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
        Returns:
        Variable Tensor
        """
        with tf.device('/cpu:0'):
            dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        return var

