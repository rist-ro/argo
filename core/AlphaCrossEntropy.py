import tensorflow as tf

from argo.core.network.AbstractModule import AbstractModule
from argo.core.utils.argo_utils import NUMTOL, tf_f1_score
from argo.core.utils.argo_utils import create_panels_lists

class AlphaCrossEntropy(AbstractModule):

    def __init__(self, multiclass_metrics = False, alpha_parameter=None ,name="CrossEntropy"):
        super().__init__(name = name)
        self._multiclass_metrics = multiclass_metrics
        #import pdb; pdb.set_trace()
        self._alpha_parameter=alpha_parameter

    @staticmethod
    def create_id(cost_fuction_kwargs):
        _id = "CE"
        alpha=cost_fuction_kwargs["alpha_parameter"]
        if alpha:
            _id +="_alp{}_".format(alpha)
        return _id

    def _build(self, model, drop_one_logit=False):

        y = model.y
        shaper = tf.shape(y)
        # if len(y.shape)==1:
        #     y = tf.expand_dims(y, axis=-1)

        kl_losses = model.kl_losses
        total_KL = tf.reduce_sum(kl_losses) / model.dataset.n_samples_train
        alpha=self._alpha_parameter
        n_samples = model.n_samples_ph

        logits = model.prediction_distr.logits
        probs = model.prediction_distr.probs
        n_labels = logits.shape[1]
        y_tile = tf.tile(y, [n_samples])
        y_true = tf.one_hot(y_tile, n_labels)





        def alphaloss(y_true, logits):
            logits_reshaped=tf.reshape(logits, (n_samples, shaper[0],n_labels))
            y_true_reshaped=tf.reshape(y_true, (n_samples, shaper[0],n_labels))
            y_true_reshaped=tf.cast(y_true_reshaped, tf.float32)
            log_solfmax=logits_reshaped-tf.reduce_max(logits_reshaped,axis=2,keepdims=True)
            log_solfmaxT=log_solfmax-tf.reduce_logsumexp(log_solfmax, axis=2,keepdims=True)
            log_cross_entropy= tf.reduce_sum(tf.multiply(y_true_reshaped,log_solfmaxT),-1)
            loss= -1./alpha*(tf.reduce_logsumexp(alpha*log_cross_entropy, 0)-tf.log(tf.cast(n_samples, tf.float32)))
            return loss

        if alpha is None:
            loss_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.cast(y_tile, tf.int32),
                                                                         logits = logits)
        else:
            loss_per_sample = alphaloss(y_true, logits)



        ce = tf.reduce_mean(loss_per_sample)
        loss = ce + total_KL

        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1),
                                                   tf.cast(y_tile, dtype = tf.int64)),
                                          dtype = tf.float32))


        # First panel will be at screen during training
        list_of_vpanels_of_plots = [
            [
                    # {
                    #     'nodes' : [loss],
                    #     'names': ["loss"],
                    #     'fileName' : "loss"
                    # },

                    {
                        'nodes': [ce],
                        'names': ["ce"],
                        'output': {'fileName': "ce"}
                    },

                    {
                        'nodes': [total_KL],
                        'names': ["kl"],
                        'output': {'fileName': "kl"}
                    },

                    {
                        'nodes': [accuracy],
                        'names': ["accuracy"],
                        'output': {'fileName': "accuracy"}
                    },

            ],
            [
                {
                    'nodes': [1 - accuracy],
                    'names': ["error"],
                    'output': {'fileName': "error", "logscale-y": 1}
                },

            ]
        ]


        # nodes_to_log = [[ce],
        #                 [total_KL],
        #                 [1 - accuracy],
        #                 [accuracy]]
        #
        # nodes_to_log_names = [["ce"], ["kl"], ["error"], ["accuracy"]]
        # nodes_to_log_filenames = [{"fileName": "ce"},
        #                           {"fileName": "kl"},
        #                           {"fileName": "error", "logscale-y": 1},
        #                           {"fileName": "accuracy"}]

        if self._multiclass_metrics:
            y_pred = tf.one_hot(tf.argmax(logits, axis=1), n_labels)
            y_true = tf.one_hot(y_tile, n_labels)
            f1_micro, f1_macro, f1_weighted = tf_f1_score(y_true, y_pred)

            auc, auc_update = tf.metrics.auc(
                                labels=tf.cast(y_true, dtype=tf.float32),
                                # predictions=tf.nn.softmax(logits)
                                predictions = probs
            )

            raise Exception("set panels correctly here first!")
            # nodes_to_log += [[auc_update],
            #                 [f1_micro, f1_macro, f1_weighted]]
            #
            # nodes_to_log_names += [["auc"], ["f1_micro", "f1_macro", "f1_weighted"]]
            # nodes_to_log_filenames += [
            #                             {"fileName": "auc"},
            #                             {"fileName": "f1_score"}
            #                             # {"fileName": "f1_micro"},
            #                             # {"fileName": "f1_macro"},
            #                             # {"fileName": "f1_weighted"}
            #                           ]



        nodes_to_log, names_of_nodes_to_log, filenames_to_log_to = create_panels_lists(list_of_vpanels_of_plots)

        return loss, loss_per_sample, nodes_to_log, names_of_nodes_to_log, filenames_to_log_to
