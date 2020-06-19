import tensorflow as tf

import pdb

import numpy as np

import pdb

from abc import ABC, abstractmethod

#from .utilsOptimizers import my_loss_full_logits

from tensorflow.python.ops.parallel_for.gradients import batch_jacobian, jacobian
        
class NaturalGradientOptimizer(tf.train.GradientDescentOptimizer):

    def __init__(self, learning_rate, *args, **kw):

        self._model = kw["model"]
        self._damping = kw["damping"]

        # remove from args before passing to the constructor of tf.train.GradientDescentOptimizer
        kw.pop("model", None)
        kw.pop("damping", None)

        if "name" not in kw.keys():
            kw["name"] = "NaturalGradient"
            
        super().__init__(learning_rate, *args, **kw)
        
    def compute_gradients(self, loss, *args, **kw):

        grads_and_vars = super().compute_gradients(loss, *args, **kw)
        
        #########################################
        # Natural gradient computed in two steps, through the Jacobian
        #########################################

        # TODO do I need to add possible regularizers to the loss
        
        logits = self._model.logits
        y = self._model.y
        #regularizer = self._model.regularizer
        # here the loss is the log-likelihood
        #loss_per_sample = self._model.loss_per_sample
        
        #n = logits.get_shape().as_list()[1]
        
        #################
        
        grads_and_vars_not_none = [(g, v) for (g, v) in grads_and_vars if g is not None]
        grads = [g for (g, v) in grads_and_vars_not_none]
        variables = [v for (g, v) in grads_and_vars_not_none]

        # no reduction of the last logit
        # previous function call
        #new_loss = my_loss_full_logits(y, logits) + regularizer
        #self.likelihood_per_sample = new_loss
        
        trainable_vars = tf.trainable_variables()

        
        
        # it was previously
        # jacobians = jacobian(self.likelihood_per_sample, trainable_vars)
        jacobians = jacobian(tf.nn.log_softmax(logits), trainable_vars) # [tf.reduce_sum(i, axis=0) for i in jacobian(self.logits, trainable_vars)]

        n_weights = int(np.sum([np.prod(i.shape[2:]) for i in jacobians])) #tf.shape(self.V)[0]
        n_samples = int(self._model.batch_size["train"]) #tf.shape(self.V)[1]

        #pdb.set_trace()
        
        self.V = tf.concat([tf.reshape(i,[tf.reduce_prod(tf.slice(tf.shape(i),[0],[2])), tf.reduce_prod(tf.slice(tf.shape(i),[2],[-1]))]) for i in jacobians], axis=1)
        self.V = tf.transpose(self.V)
        self.Q = tf.reshape(tf.nn.softmax(logits), [-1])
               
        # this is an expected value, so I need to divide by the number of samples
        #self.V /= n_samples
        self.Q /= n_samples

        damp = self._damping
                
        #K = tf.einsum('ki,kj->ij', self.V, self.V)
        # faster
        K = 1/damp*tf.matmul(self.V, self.V, transpose_a = True)
        
        #D = tf.eye(n_weights)

        # verify invFisher
        # I = np.eye(3)
        # D = np.eye(83)
        # D/damp - 1/damp/damp * np.dot(np.dot(V,np.linalg.inv(I + 1/damp*np.dot(V.T,V))),V.T)
        
        G = tf.concat([tf.reshape(g,[tf.reduce_prod(tf.shape(g)), 1]) for g in grads], axis=0)

        #I = tf.eye(n_samples)
        S = tf.linalg.inv(tf.diag(1/self.Q) + K)
        #self.VS = tf.einsum('ik,kj->ij', self.V, S)
        # faster
        self.VS = tf.matmul(self.V, S)
        #self.invFisher = 1/damp*D - 1/damp**2*tf.einsum('ik,jk->ij', self.VS, self.V)

        ############### not efficient since I compuute the inverse matrix
        # faster due to matmul
        #self.invFisher = 1/damp*D - 1/damp**2*tf.matmul(self.VS, self.V, transpose_b = True)
        #self.NG = tf.einsum('ij,jk->ik', self.invFisher, G)
        # faster due to matmul
        #self.NG = tf.matmul(self.invFisher, G)
        ###############

        size = n_samples * self._model.dataset.n_labels 

        self._invFisher_D = 1/damp*tf.ones(shape=[n_weights, 1])
        self._invFisher_S = -tf.reshape(S, [size, size]) 
        self._invFisher_V = 1/damp*tf.reshape(self.V,[n_weights, size])
        self._n_weights_layers = [int(np.prod(i.shape[2:])) for i in jacobians]
        
        # much faster!
        self.NG = 1/damp*G - (1/damp)**2*tf.matmul(self.VS, tf.matmul(self.V,
                                                                      G,
                                                                      transpose_a = True))
        
        #self._model.loss = tf.reduce_mean(new_loss)

        # remove this line
        #self.grads = ng
            
        self.natural_gradient = []
        start = 0
        for i in range(len(grads)):
            length = tf.reduce_prod(tf.shape(grads[i]))
            c = tf.slice(self.NG, [start, 0], [length, 1])
            start += length
            d = tf.reshape(c, grads[i].shape)
            self.natural_gradient.append(d)

        # restore the gradient
        grads_and_vars = [(g, v) for (g, v) in zip(self.natural_gradient, variables)]

        return grads_and_vars


    ##########################################
    # start experiments for natural gradient #
    ##########################################
    '''
    l = []
    for i in range(len(self.grads_and_vars)):
        a = tf.reshape(self.grads_and_vars[i][0],[-1,])
        l.append(a)
        print(a)
        b = tf.reshape(a, self.grads_and_vars[i][0].shape)
        print(b)

    # here I need to comute the Fisher information matrix and invert it
    g = tf.concat(l,axis=0)

    # here I should compute the natural gradient

    start = 0
    for i in range(len(self.grads_and_vars)):
        #pdb.set_trace()
        length = l[i].get_shape().as_list()[0]
        c = tf.slice(g, [start], [length])
        start += length
        print(c)
        d = tf.reshape(c, self.grads_and_vars[i][0].shape)
        print(d)

    # apply the gradients
    '''

    ##########################################
    # end experiments for natural gradient #
    ##########################################
