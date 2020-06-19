import tensorflow as tf
from argo.core.network.AbstractModule import AbstractModule
from argo.core.utils.argo_utils import create_panels_lists

class AlphaLikelihood(AbstractModule):

    """
    This is Alpha Likelihood for a regression problem
    """

    def __init__(self, alpha_parameter=None, n_samples=1, name="LL"): # drop_one_logit=0,
        super().__init__(name = name)
        self._alpha_parameter = alpha_parameter
        self._use_alpha = (alpha_parameter is not None)

    @staticmethod
    def create_id(cost_fuction_kwargs):
        alpha_parameter=cost_fuction_kwargs.get('alpha_parameter')
        use_alpha=cost_fuction_kwargs.get('use_alpha')
        n_samples=cost_fuction_kwargs.get('n_samples')
        _id = "LL-"
        if use_alpha:
            _id+="alpha:{0:.2g}".format(alpha_parameter)
            _id+="-asam:{0:.1g}".format(n_samples)#+ str(cost_fuction_kwargs.get("drop_one_logit",0))
        return _id

    def _build(self, model): #, drop_one_logit=False):

        n_samples=model.n_samples_ph
        y = model.y
        shaper = tf.shape(y)
        distr = model.prediction_distr

        if self._use_alpha:
            
            if self._alpha_parameter!=0:
                y_tile = tf.tile(y, [n_samples, 1])
                loss_core = -distr.log_prob(y_tile)
                #loss_per_minibatch = tf.exp(tf.scalar_mul(self._alpha_parameter,distr.log_prob(y_tile)))
                #loss_per_minibatch_reshaped=tf.reshape(loss_per_minibatch, (alpha_samples,shaper[0]))
                #loss_per_minibatch_avg=tf.reduce_mean(loss_per_minibatch_reshaped,axis=0)
                #loss_per_sample=tf.scalar_mul(-1./self._alpha_parameter,tf.log(loss_per_minibatch_avg))
                loss_per_minibatch = tf.scalar_mul(self._alpha_parameter,distr.log_prob(y_tile))
                #import pdb; pdb.set_trace()
                loss_per_minibatch_reshaped=tf.reshape(loss_per_minibatch, (n_samples, shaper[0]))
                loss_per_minibatch_avg=tf.reduce_logsumexp(loss_per_minibatch_reshaped,axis=0)
                loss_per_sample=tf.scalar_mul(-1./self._alpha_parameter,loss_per_minibatch_avg)
            else:
                y_tile = tf.tile(y, [n_samples, 1])
                loss_core = -distr.log_prob(y_tile)
                loss_per_minibatch = -distr.log_prob(y_tile)
                loss_per_minibatch_reshaped=tf.reshape(loss_per_minibatch, (n_samples, shaper[0]))
                loss_per_sample=tf.reduce_mean(loss_per_minibatch_reshaped, axis=0)

        else:
            loss_per_sample = -distr.log_prob(y)
            loss_core = loss_per_sample

        nll = tf.reduce_mean(loss_per_sample, name="nll")
        kl_losses = model.kl_losses
        total_KL = tf.reduce_sum(kl_losses) / model.dataset.n_samples_train
        loss = nll + total_KL
        nll_core = tf.reduce_mean(loss_core, name="nll_core")

        # in case of Bayesian network I need to add kl_losses for the weights if I want to see them
        # (otherwise kl_losses will be an empty list for non bayesian predictions)
        # if kl_losses:
        #
        #     KL_i_names = ["KL_" + str(int(i+1)) for i, l in enumerate(kl_losses)]
        #
        #     nodes_to_log = [[loss],
        #                     [nll],
        #                     # [total_KL],
        #                     # kl_losses
        #                     ]
        #
        #     names_of_nodes_to_log = [["loss"],
        #                              ["NLL"],
        #                              # ["total_KL"],
        #                              # KL_i_names
        #                              ]
        #
        #     filenames_to_log_to = [{"fileName" : "loss"},
        #                             {"fileName" : "negloglikelihood"},
        #                             # {"fileName" : "total_KL"},
        #                             # {"fileName" : "all_KLs", "legend": 0}
        #                            ]
        #
        # else:


        means = model.prediction_mean
        # if self._use_alpha:
        means=tf.reshape(means, (n_samples,shaper[0],shaper[1]))
        means=tf.reduce_mean(means,axis=0)
        # else:
        #     pass

        mse_per_sample = tf.reduce_sum(tf.square(y - means), axis=1)
        mse = tf.reduce_mean(mse_per_sample)

        # First panel will be at screen during training
        list_of_vpanels_of_plots = [
            [
                    {
                        'nodes' : [loss],
                        'names': ["loss"],
                        'output': {'fileName' : "loss"}
                    },

                    {
                        'nodes': [nll],
                        'names': ["NLL"],
                        'output': {'fileName': "negloglikelihood"}
                    },

                    {
                        'nodes': [mse],
                        'names': ["mse"],
                        'output': {'fileName': "mse"}
                    }
            ]
        ]

        nodes_to_log, names_of_nodes_to_log, filenames_to_log_to = create_panels_lists(list_of_vpanels_of_plots)

        # nodes_to_log = [[loss], [nll], [mse], [loss_core]]
        #
        # names_of_nodes_to_log = [["loss"], ["NLL"], ["MSE"], ["loss_core"]]
        #
        # filenames_to_log_to = [{"fileName": "loss"},
        #                        {"fileName": "negloglikelihood"},
        #                        {"fileName": "mse"},
        #                        {"fileName": "loss_core"}
        #                        ]

        return loss, loss_per_sample, nodes_to_log, names_of_nodes_to_log, filenames_to_log_to
