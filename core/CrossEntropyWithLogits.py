import tensorflow as tf

from argo.core.network.AbstractModule import AbstractModule
from argo.core.utils.argo_utils import NUMTOL, tf_f1_score
from argo.core.utils.argo_utils import create_panels_lists


class CrossEntropyWithLogits(AbstractModule):

    def __init__(self, multiclass_metrics = False, nbp = False, name="CrossEntropyWithLogits"): # drop_one_logit=0,
        super().__init__(name = name)
        self._multiclass_metrics = multiclass_metrics
        self._nbp = nbp
        #self._drop_one_logit = drop_one_logit
        
    def create_id(self, cost_fuction_kwargs):
        _id = "CE" #+ str(cost_fuction_kwargs.get("drop_one_logit",0))
        if cost_fuction_kwargs.get("nbp", False):
            _id += "_nbp"

        return _id

    def _build(self, prediction_model): #, drop_one_logit=False):

        y = prediction_model.y
        logits = prediction_model.logits
        n_labels = logits.shape[1]


        # do not delete yet (Luigi)
        '''
        if drop_one_logit:
            n = logits.get_shape().as_list()[1]
        else:
        '''

        loss_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.cast(y, tf.int32),
                                                                         logits = logits)
        if self._nbp:

            @tf.custom_gradient
            def nbp(x):
                #pdb.set_trace()
                def grad(dy):
                    I_theta = tf.stop_gradient(tf.nn.softmax(logits))
                    
                    #tf.tensordot(tf.linalg.inv(I_theta),dy, axes=0) #tf.dot(tf.linalg.inv(I_theta),dy)
                    #print_op = tf.Print(tf.shape(dy), [tf.shape(dy)])
                    #with tf.control_dependencies([print_op]):
                    #    return dy #tf.matmul(tf.diag(tf.linalg.inv(I_theta)), dy)
                    #return tf.matmul(tf.linalg.inv(I_theta), dy)

                    pdb.set_trace()
                    
                    return tf.matmul(I_theta, dy)
                
                return tf.identity(x), grad

            loss_per_sample = nbp(loss_per_sample)

        loss = tf.reduce_mean(loss_per_sample)

        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1),
                                                   tf.cast(y, dtype = tf.int64)),
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
                        'nodes': [loss],
                        'names': ["ce"],
                        'output': {'fileName': "ce"}
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


        # nodes_to_log = [[1 - accuracy],
        #                 [accuracy]]
        # nodes_to_log_names = [["err"], ["acc"]]
        # nodes_to_log_filenames = [{"fileName": "error", "logscale-y": 1},
        #                           {"fileName": "accuracy"}]


        if self._multiclass_metrics:
            y_pred = tf.one_hot(tf.argmax(logits, axis=1), n_labels)
            y_true = tf.one_hot(y, n_labels)
            f1_micro, f1_macro, f1_weighted = tf_f1_score(y_true, y_pred)

            auc, auc_update = tf.metrics.auc(
                                labels=tf.cast(y_true, dtype=tf.float32),
                                predictions=tf.nn.softmax(logits)
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

    
