from .ArgoAbstractKerasNetwork import ArgoAbstractKerasNetwork
from ..keras_models.keras_utils import parse_init_reg_kwargs, parse_init_reg_kwargs_bayes
from ..utils.argo_utils import get_method_id
import importlib
from pprint import pprint

class KerasNetwork(ArgoAbstractKerasNetwork):

    default_params = {
        **ArgoAbstractKerasNetwork.default_params,

        "init_reg_kwargs" : {

            "kernel_initializer": None,
            "bias_initializer": None,

            "kernel_regularizer": ("contrib.layers.l2_regularizer", {"scale": 1e-5}),
            "bias_regularizer": ("contrib.layers.l2_regularizer", {"scale": 1e-5}),

            "activity_regularizer": None,
        },


        "init_reg_kwargs_bayes": {

            "posterior":
                {
                    "kernel_untr_scale_initializer": ("initializers.random_normal", {"mean": -9., "stddev": 1e-2}),
                    "kernel_loc_initializer": ("initializers.random_normal", {"mean": 0., "stddev": 1e-2}),
                    "bias_loc_initializer": ("initializers.constant", {"value": 1.}),

                    "kernel_loc_regularizer": ("contrib.layers.l2_regularizer", {"scale": 1e-5}),
                    "kernel_untr_scale_regularizer": ("contrib.layers.l2_regularizer", {"scale": 1e-5}),
                    "bias_loc_regularizer": ("contrib.layers.l2_regularizer", {"scale": 0.}),

                    "activity_regularizer": None,
                },

            "prior":
                {
                    "kernel_loc_initializer": ("initializers.constant", {"value": 0.}),
                    "kernel_untr_scale_initializer": ("initializers.constant", {"value": 0.5}),
                    "trainable" : False,
                    "default" : False
                }
        }
    }

    def create_id(self):
        opts = self._opts
        _id = ""
        if opts["init_reg_kwargs_bayes"] is not None:
            _id += "_LI" + get_method_id(opts["init_reg_kwargs_bayes"]["posterior"]["kernel_loc_initializer"])
            _id += "_" + get_method_id(opts["init_reg_kwargs_bayes"]["posterior"]["bias_loc_initializer"])
            _id += "_klr" + get_method_id(opts["init_reg_kwargs_bayes"]["posterior"]["kernel_loc_regularizer"])
            _id += "_ksr" + get_method_id(opts["init_reg_kwargs_bayes"]["posterior"]["kernel_untr_scale_regularizer"])
            _id += "_blr" + get_method_id(opts["init_reg_kwargs_bayes"]["posterior"]["bias_loc_regularizer"])
            _id += "_SCM" + str(opts["init_reg_kwargs_bayes"]["posterior"]["kernel_untr_scale_constraint_max"])

            if opts["init_reg_kwargs_bayes"]["prior"]["default"]:
                _id+= "_Pdef"
            else:
                _id += "_PS" + get_method_id(opts["init_reg_kwargs_bayes"]["prior"]["kernel_untr_scale_initializer"])
                _id += "_tr" + str(int(opts["init_reg_kwargs_bayes"]["prior"]["trainable"]))

        if opts["init_reg_kwargs"] is not None:
            _id += "_ki" + get_method_id(opts["init_reg_kwargs"]["kernel_initializer"])
            _id += "_bi" + get_method_id(opts["init_reg_kwargs"]["bias_initializer"])
            _id += "_kr" + get_method_id(opts["init_reg_kwargs"]["kernel_regularizer"])
            _id += "_br" + get_method_id(opts["init_reg_kwargs"]["bias_regularizer"])

        # if opts["init_reg_kwargs"] is not None:
        #     _id += "-ki" + get_method_id(opts["init_reg_kwargs"]["kernel_initializer"])
        #     _id += "-bi" + get_method_id(opts["init_reg_kwargs"]["bias_initializer"])
        #     _id += "-kr" + get_method_id(opts["init_reg_kwargs"]["kernel_regularizer"])
        #     _id += "-br" + get_method_id(opts["init_reg_kwargs"]["bias_regularizer"])

        super_id = super().create_id()

        _id += super_id

        return _id

    def __init__(self, opts, name=None):
        super().__init__(opts, name)
        self._layer_kwargs = parse_init_reg_kwargs(opts["init_reg_kwargs"])
        self._layer_kwargs_bayes = parse_init_reg_kwargs_bayes(opts["init_reg_kwargs_bayes"])



    def get_keras_losses(self, inputs):
        net_reg_losses, net_kl_losses, net_update_ops, net_all_layers = self._get_rkua_lists(self._net_model, inputs)
        distr_reg_losses, distr_kl_losses, distr_update_ops, distr_all_layers = self._get_rkua_lists(self._distr_model, inputs)

        all_layers = distr_all_layers + net_all_layers
        reg_losses = distr_reg_losses + net_reg_losses
        kl_losses = distr_kl_losses + net_kl_losses
        update_ops = distr_update_ops + net_update_ops

        # reg_losses, kl_losses, update_ops, all_layers = self._get_rkua_lists(self._model, inputs)

        # pprint(all_layers)
        pprint([l.name for l in all_layers])
        print("")
        # batch_norm_num = len([l for l in all_layers if "batch_norm" in l.name.lower()])
        # flipout_num = len([l for l in all_layers if "flipout" in l.name.lower()])
        # reparameterization_num = len([l for l in all_layers if "reparameterization" in l.name.lower()])
        # print("found {} BatchNorm layers".format(batch_norm_num))
        # print("found {} Flipout layers".format(flipout_num))
        # print("found {} Reparameterization layers".format(reparameterization_num))

        print("")
        print("found {} keras regularizers".format(len(reg_losses)))
        pprint(reg_losses)

        print("")
        print("found {} keras kl losses".format(len(kl_losses)))
        pprint(kl_losses)

        print("")
        print("found {} keras update ops".format(len(update_ops)))
        pprint(update_ops)

        return reg_losses, kl_losses, update_ops


    def _get_rkua_lists(self, model, inputs):
        if model is None:
            return [], [], [], []

        losses = model.get_losses_for(None) + model.get_losses_for(inputs)
        reg_losses = [l for l in losses if "regularizer" in l.name.lower()]
        kl_losses = [l for l in losses if "divergence" in l.name.lower()]
        update_ops = model.get_updates_for(None) + model.get_updates_for(inputs)
        print("")
        model.summary()
        print("")

        print("found {:} layers for model {:}\n".format(len(model.layers), model.name))
        all_layers = model.layers

        return reg_losses, kl_losses, update_ops, all_layers

    #
    # def _keras_model_builder_from_input(self, input_shape):
    #     inputs = tf.keras.layers.Input(shape=input_shape, dtype='float32')
    #
    #     self._net_model = self._keras_model_builder(self._keras_model_name,
    #                                                 self._keras_model_kwargs)
    #
    #     self._distr_model = self._keras_model_builder(self._distribution_name,
    #                                                   self._distribution_kwargs)
    #
    #     logits = self._net_model(inputs)
    #     outputs = self._distr_model(logits)
    #
    #     model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self._keras_model_name)
    #     return model


    def _keras_model_builder(self, keras_model_name, keras_model_kwargs):
        # load the keras model specified
        keras_pymodule = importlib.import_module("..keras_models." + keras_model_name,
                                               '.'.join(__name__.split('.')[:-1]))

        make_model = getattr(keras_pymodule, keras_model_name)

        model = make_model(**keras_model_kwargs)

        return model
