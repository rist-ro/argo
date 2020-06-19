from argo.core.network.KerasNetwork import KerasNetwork
from argo.core.flows.build_flow import build_flow
import tensorflow_probability as tfp

class PredictionKerasNetwork(KerasNetwork):
    """
    Network class for managing the network used for CMB
    """

    default_params = {
        **KerasNetwork.default_params,

        "network_architecture" : [
            [("bayesian_vgg", {}),
             ("MultivariateNormalDiag", {"bayesian_layers" : True})]
        ],

    }

    def create_id(self):

        # keras_model_tuple, distr_tuple, flow_params = self._parse_network_architecture(opts["network_architecture"])
        # #delegate id creation to keras_models utils
        # _id = "-n" + get_network_id(keras_model_tuple)
        # if distr_tuple is not None:
        #     _id += "_" + get_network_id(distr_tuple)
        #
        #     if flow_params is not None:
        #         _id += "-f" + flow_params['name']
        #         _id += "_n{:d}".format(flow_params['num_bijectors'])
        #         _id += "_hc{:d}".format(flow_params['hidden_channels'])

        #delegate id creation to subunits
        _id = "-n" + self._net_model._id()

        if self._distr_model is not None:
            _id += "_" + self._distr_model._id()

            if self._flow is not None:
                #TODO make flow a class too
                _id += "-f" + self._flow_params['name']
                _id += "_n{:d}".format(self._flow_params['num_bijectors'])
                _id += "_hc{:d}".format(self._flow_params['hidden_channels'])

        super_id = super().create_id()

        _id += super_id
        return _id

    def _parse_network_architecture(self, network_architecture):
        # this can be 2 or 3
        if len(network_architecture) == 2:
            keras_model_tuple, distribution_tuple = network_architecture
            flow_params = None
        elif len(network_architecture) == 3:
            keras_model_tuple, distribution_tuple, flow_params = network_architecture
        else:
            raise Exception("`network_architecture` can be composed by either 2 or 3 tuples.")
        return keras_model_tuple, distribution_tuple, flow_params

    def __init__(self, opts, name=None):
        if name is None:
            name = self._keras_model_name

        super().__init__(opts, name)

        self._network_architecture = opts["network_architecture"]

        keras_model_tuple, distribution_tuple, flow_params = self._parse_network_architecture(self._network_architecture)

        self._keras_model_name, self._keras_model_kwargs = keras_model_tuple
        self._keras_model_kwargs.update({
            "layer_kwargs" : self._layer_kwargs,
            "layer_kwargs_bayes" : self._layer_kwargs_bayes
        })

        # activation = get_activation(opts)
        if len(self._opts["output_shape"])>1:
            raise ValueError("Not implemented prediction with non scalar outputs")

        self._input_shape = opts["input_shape"]

        self._prob_bool = (distribution_tuple is not None)

        if self._prob_bool:
            self._distribution_name, self._distribution_kwargs = distribution_tuple
            self._distribution_kwargs.update({
                "layer_kwargs" : self._layer_kwargs,
                "layer_kwargs_bayes" : self._layer_kwargs_bayes
            })


        self._try_set_output_shape(opts)

        self._flow_params = flow_params

        # with self._enter_variable_scope():
        self._net_model = self._keras_model_builder(self._keras_model_name,
                                                    self._keras_model_kwargs,
                                                    )

        self._distr_model = None
        self._flow = None

        if self._prob_bool:
            self._distr_model = self._keras_model_builder(self._distribution_name,
                                                          self._distribution_kwargs)

            # FLOW
            if self._flow_params is not None:
                flow_size = self._output_size
                self._flow = build_flow(self._flow_params, flow_size)

        # self._model = self._keras_model_builder_from_input(self._input_shape)

    def _try_set_output_shape(self, opts):
        # in classification case we keep only a scalar int.. if we would keep y as one_hot there would not be this problem

        # CHECK IF OUTPUT_SHAPE HAS BEEN SET MANUALLY
        if self._prob_bool:
            has_been_set = (self._distribution_kwargs.get("output_size", None) is not None)
        else:
            has_been_set = (self._keras_model_kwargs.get("output_size", None) is not None)

        if not has_been_set:
            # TRY SET OUTPUT SHAPE
            self._output_size = opts["output_shape"][0]
            if self._prob_bool:
                self._distribution_kwargs.update({"output_size": self._output_size})
            else:
                self._keras_model_kwargs.update({"output_size": self._output_size})


    def call(self, inputs, is_training=False, **extra_kwargs):
        logits = self._net_model(inputs, training=is_training)

        if self._prob_bool:
            outputs = self._distr_model(logits, training=is_training)

            if self._flow is not None:
                outputs = tfp.distributions.TransformedDistribution(
                                                    distribution=outputs,
                                                    bijector=self._flow)

        else:
            outputs = logits

        # TODO is it possible to merge everything in a single module with the same flexibility?
        # outputs = self._model(inputs, training=is_training)
        return outputs
