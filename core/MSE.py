import tensorflow as tf
from argo.core.network.AbstractModule import AbstractModule
from argo.core.utils.argo_utils import create_panels_lists

class MSE(AbstractModule):

    def __init__(self, name="MSE"): # drop_one_logit=0, 
        super().__init__(name = name)

        #self._drop_one_logit = drop_one_logit
        
    @staticmethod
    def create_id(cost_fuction_kwargs):

        _id = "MSE" #+ str(cost_fuction_kwargs.get("drop_one_logit",0))
                
        return _id

    def _build(self, prediction_model, drop_one_logit=False):

        y = prediction_model.y
        logits = prediction_model.logits

        loss_per_sample = tf.reduce_mean(tf.square(y - logits), axis=1)
        loss = tf.reduce_mean(loss_per_sample)

        # First panel will be at screen during traininig
        list_of_vpanels_of_plots = [
            [
                    {
                        'nodes' : [loss],
                        'names': ["mse"],
                        'output': {'fileName' : "mse"}
                    },
            ]
        ]

        nodes_to_log, names_of_nodes_to_log, filenames_to_log_to = create_panels_lists(list_of_vpanels_of_plots)

        return loss, loss_per_sample, nodes_to_log, names_of_nodes_to_log, filenames_to_log_to

    
