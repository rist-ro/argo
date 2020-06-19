import tensorflow as tf

from argo.core.network.AbstractModule import AbstractModule


class Likelihood(AbstractModule):

    def __init__(self, name="LL"): # drop_one_logit=0,
        super().__init__(name = name)

        #self._drop_one_logit = drop_one_logit

    @staticmethod
    def create_id(cost_fuction_kwargs):

        _id = "LL" #+ str(cost_fuction_kwargs.get("drop_one_logit",0))

        return _id

    def _build(self, model): #, drop_one_logit=False):

        x_target = model.x_target
        distr = model._model_visible

        # if it is AE it does not have n_z_samples so it defaults to one and does not replicate
        n_z_samples = getattr(model, "n_z_samples", 1)
        input_shape = x_target.shape.as_list()[1:]
        ones = [1] * len(input_shape)
        x_replicate = tf.tile(x_target, [n_z_samples] + ones)

        reconstr_loss = -distr.log_prob(x_replicate)

        all_axis_but_first = list(range(len(reconstr_loss.shape)))[1:]
        # independent p for each input pixel
        loss_per_sample = tf.reduce_sum(reconstr_loss, axis=all_axis_but_first)
        # average over all the samples and the batch (they are both stacked on the axis 0)
        loss = tf.reduce_mean(loss_per_sample, axis=0, name="nll")

        if len(loss.shape)>0:
            raise RuntimeError("loss should be a scalar at this point, found shape: {}".format(loss.shape))

        nodes_to_log = [[-loss]]
        names_of_nodes_to_log = [['LL']]
        filenames_to_log_to = [{"fileName" : "likelihood"}]

        return loss, nodes_to_log, names_of_nodes_to_log, filenames_to_log_to