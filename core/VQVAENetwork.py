import tensorflow as tf
import sonnet as snt
import tensorflow_probability as tfp

from argo.core.network.ArgoStochasticNetworkWithDefaults import ArgoStochasticNetworkWithDefaults
from argo.core.network.GeneralSonnetNetwork import GeneralSonnetNetwork
from argo.core.utils import argo_utils as utils


class VQVAENetwork(ArgoStochasticNetworkWithDefaults):
    """
    Network class for managing a VAE network
    """

    # if not present in opts, the values specified in default_params will be provided automagically
    # see more in AbstractNetwork.__init__
    default_params = {
        **ArgoStochasticNetworkWithDefaults.default_params,
        "network_architecture" : {

            "encoder" : [
                ("Linear", {"output_size" : 200}, 1),
                ("Linear", {"output_size" : 200}, 1)],

            "vq" : {"pre" : "Linear", # Conv2D
                    "embedding_dim" : 64,
                    "latent_channels": 1,
                    "num_embeddings": 512,
                    "commitment_cost" : 0.25,
                    "prior" : "fixed", #"train"
                    },

            "decoder" : [
                ("Linear", {"output_size" : 200}, 1),
                ("Linear", {"output_size" : 200}, 1),
                ("GaussianDiagonalZeroOne", {"minimal_covariance" : 0}, 0)],

        },

        "samples" : 10,
    }

    def create_id(self):

        encoder_layers_ids = filter(None,
                                    [utils.get_method_id(layer_tuple)
                                        for layer_tuple in self._opts["network_architecture"]["encoder"]]
                             )

        decoder_layers_ids = filter(None,
                                    [utils.get_method_id(layer_tuple)
                                        for layer_tuple in self._opts["network_architecture"]["decoder"]]
                              )

        _id = '-ne_' + "_".join(encoder_layers_ids) +\
              '-vq_' + "e{:}".format(self._opts["network_architecture"]["vq"]["embedding_dim"]) + \
              "_p" + self._opts["network_architecture"]["vq"]["pre"][0] + \
              "_lc{:}".format(self._opts["network_architecture"]["vq"]["latent_channels"]) + \
              "_c{:}".format(self._opts["network_architecture"]["vq"]["commitment_cost"]) + \
              "_pr" + self._opts["network_architecture"]["vq"]["prior"][0].upper() + \
              '-nd_' + "_".join(decoder_layers_ids)


        super_id = super().create_id()

        _id += super_id
        return _id


    def __init__(self, opts, name="vae_network", seed=None):
        """Short summary.

        Args:
            opts (dict): parameters of the task.
            name (str): name of the Sonnet module.
        """

        super().__init__(opts, name, seed)

        network_architecture = self._opts["network_architecture"]
        # TODO-ARGO2 y_name, self._pre_kwargsou might want to check here that the 2 architectures (encoder and decoder) expected are effectively passed
        self._network_architecture = network_architecture
        self._vq_kwargs = network_architecture["vq"]
        self._embedding_dim = self._vq_kwargs["embedding_dim"]
        self._num_embeddings = self._vq_kwargs["num_embeddings"]
        self._commitment_cost = self._vq_kwargs["commitment_cost"]
        self._latent_channels = self._vq_kwargs["latent_channels"]

        self._pre_name = self._vq_kwargs["pre"]

        self._prior_tag = self._vq_kwargs["prior"]
        assert self._prior_tag in ["fixed", "train"], "prior `{:}` not recognized".format(self._prior_tag)

        self.n_samples_prior = tf.placeholder_with_default(1, shape=(), name='n_samples_prior')


    def _build(self, x, is_training, network_str=""):
        """
        Args:
            x (tf.tensor): input node.
            network_str (str): Optional network_str specifying the network we are going to build.
                            It is used to set some specific collections for activity and contractive regularizers.

        Returns:
            tf distribution of the latent space
            tf distribution on the visible neurons
        """

        # ENCODER
        self.encoder_module = GeneralSonnetNetwork(self._network_defaults["activation"],
                                                   self._network_defaults["weights_init"],
                                                   self._network_defaults["bias_init"],
                                                   self._network_defaults["weights_reg"],
                                                   self._network_defaults["bias_reg"],
                                                   self._network_architecture["encoder"],
                                                   self._stochastic_defaults,
                                                   is_training=is_training,
                                                   network_str=network_str,
                                                   name="encoder")

        # LINKING
        pre_output_channels = self._embedding_dim * self._latent_channels
        if self._pre_name=="Conv2D":
            pre_vq_layer = snt.Conv2D(output_channels=pre_output_channels,
                                      kernel_shape=(1, 1),
                                      stride=(1, 1),
                                      name="to_vq")
        elif self._pre_name=="Linear":
            pre_vq_layer = snt.Linear(output_size=pre_output_channels,
                                      name="to_vq")
        else:
            raise Exception("pre vq layer not recognized: `{:}`".format(self._pre_name))

        out_pre = pre_vq_layer(self.encoder_module(x))

        z_preshape = [-1] + out_pre.shape[1:].as_list()
        z_shape = z_preshape[:-1] + [self._latent_channels, self._embedding_dim]
        self.z_e = tf.reshape(out_pre, z_shape)

        # VQ
        self.vq_layer = VectorQuantizer(
                            embedding_dim=self._embedding_dim,
                            num_embeddings=self._num_embeddings,
                            commitment_cost=self._commitment_cost)

        vq_out = self.vq_layer(self.z_e, is_training=is_training)

        self._cat_posterior = vq_out["encoding_distr"]
        self.z_q = vq_out["quantize"]

        # I set the shape of the visible layer, to match the input shape
        input_shape = x.get_shape().as_list()[1:]
        self._network_architecture["decoder"][-1][1]["output_shape"] = input_shape
        # 4) and 5)
        self.decoder_module = GeneralSonnetNetwork(self._network_defaults["activation"],
                                                   self._network_defaults["weights_init"],
                                                   self._network_defaults["bias_init"],
                                                   self._network_defaults["weights_reg"],
                                                   self._network_defaults["bias_reg"],
                                                   self._network_architecture["decoder"],
                                                   self._stochastic_defaults,
                                                   is_training=is_training,
                                                   network_str=network_str,
                                                   name="decoder")

        self._model_visible = self.decoder_module(tf.reshape(self.z_q, z_preshape))

        self._embeddings = self.vq_layer.embeddings
        # create prior
        with tf.variable_scope('generation'):

            # uniform prior, not the best option in general, this can be learned afterwards
            self._prior_vars = tf.get_variable("prior_vars",
                                               shape=(self._num_embeddings,),
                                               initializer=tf.constant_initializer(1.),
                                               trainable= True if self._prior_tag=="train" else False
                                            )
            self._cat_prior = tfp.distributions.Categorical(logits=self._prior_vars, name='prior_cat')

            # independent samples, this can be improved
            batch_shape = [self.n_samples_prior] + self.z_q.shape[1:-1].as_list()
            cat_samples = self._cat_prior.sample(batch_shape)

            quantized_samples = self.vq_layer.quantize(cat_samples)
            self._prior_samples = tf.reshape(quantized_samples, z_preshape)

            # attach the decoder to the samples
            samples = self.decoder_module(self._prior_samples)


        out_nodes = {
                        'z_e': self.z_e,
                        'z_q': self.z_q,
                        'posterior' : self._cat_posterior,
                        'encodings': vq_out['encodings'],
                        'encoding_indices': vq_out['encoding_indices'],
                        'embeddings' : self._embeddings,
                        'reconstruction_model' : self._model_visible,
                        'prior' : self._cat_prior,
                        'generation_node' : samples.reconstruction_node()
                    }

        net_losses = {
                        'vq_loss': vq_out['loss'],
                        'perplexity': vq_out['perplexity']
        }

        return out_nodes, net_losses




# from sonnet, modified
class VectorQuantizer(snt.AbstractModule):
  """Sonnet module representing the VQ-VAE layer.

  Implements the algorithm presented in
  'Neural Discrete Representation Learning' by van den Oord et al.
  https://arxiv.org/abs/1711.00937

  Input any tensor to be quantized. Last dimension will be used as space in
  which to quantize. All other dimensions will be flattened and will be seen
  as different examples to quantize.

  The output tensor will have the same shape as the input.

  For example a tensor with shape [16, 32, 32, 64] will be reshaped into
  [16384, 64] and all 16384 vectors (each of 64 dimensions)  will be quantized
  independently.

  Args:
    embedding_dim: integer representing the dimensionality of the tensors in the
      quantized space. Inputs to the modules must be in this format as well.
    num_embeddings: integer, the number of vectors in the quantized space.
    commitment_cost: scalar which controls the weighting of the loss terms
      (see equation 4 in the paper - this variable is Beta).
  """

  def __init__(self, embedding_dim, num_embeddings, commitment_cost,
               name='vq_layer'):
    super(VectorQuantizer, self).__init__(name=name)
    self._embedding_dim = embedding_dim
    self._num_embeddings = num_embeddings
    self._commitment_cost = commitment_cost

    with self._enter_variable_scope():
      initializer = tf.uniform_unit_scaling_initializer()
      self._w = tf.get_variable('embedding', [embedding_dim, num_embeddings],
                                initializer=initializer, trainable=True)

  def _build(self, inputs, is_training):
    """Connects the module to some inputs.

    Args:
      inputs: Tensor, final dimension must be equal to embedding_dim. All other
        leading dimensions will be flattened and treated as a large batch.
      is_training: boolean, whether this connection is to training data.

    Returns:
      dict containing the following keys and values:
        quantize: Tensor containing the quantized version of the input.
        loss: Tensor containing the loss to optimize.
        perplexity: Tensor containing the perplexity of the encodings.
        encodings: Tensor containing the discrete encodings, ie which element
          of the quantized space each input element was mapped to.
        encoding_indices: Tensor containing the discrete encoding indices, ie
          which element of the quantized space each input element was mapped to.
    """
    # Assert last dimension is same as self._embedding_dim
    input_shape = tf.shape(inputs)

    with tf.control_dependencies([
        tf.Assert(tf.equal(input_shape[-1], self._embedding_dim),
                  [input_shape])]):
      flat_inputs = tf.reshape(inputs, [-1, self._embedding_dim])

    distances = (tf.reduce_sum(flat_inputs**2, 1, keepdims=True)
                 - 2 * tf.matmul(flat_inputs, self._w)
                 + tf.reduce_sum(self._w ** 2, 0, keepdims=True))

    logits_shape = tf.concat([tf.shape(inputs)[:-1], [self._num_embeddings]], axis=0)
    encoding_cat_distr = tfp.distributions.Categorical(logits=tf.reshape(-distances, logits_shape), name='posterior_cat')

    encoding_indices = tf.argmax(- distances, 1)
    encodings = tf.one_hot(encoding_indices, self._num_embeddings)
    encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])
    quantized = self.quantize(encoding_indices)

    e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)
    q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)
    loss = q_latent_loss + self._commitment_cost * e_latent_loss

    quantized = inputs + tf.stop_gradient(quantized - inputs)
    avg_probs = tf.reduce_mean(encodings, 0)
    perplexity = tf.exp(- tf.reduce_sum(avg_probs * tf.log(avg_probs + 1e-10)))

    return {'quantize': quantized,
            'loss': loss,
            'perplexity': perplexity,
            'encodings': encodings,
            'encoding_indices': encoding_indices,
            'encoding_distr' : encoding_cat_distr}

  @property
  def embeddings(self):
    return self._w

  def quantize(self, encoding_indices):
    with tf.control_dependencies([encoding_indices]):
      w = tf.transpose(self.embeddings.read_value(), [1, 0])
    return tf.nn.embedding_lookup(w, encoding_indices, validate_indices=False)

