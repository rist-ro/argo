import tensorflow as tf

from argo.core.network.AbstractModule import AbstractModule


class L2(AbstractModule):

    def __init__(self, name="L2"):
        super().__init__(name = name)
        
    @staticmethod
    def create_id(cost_fuction_kwargs):
        return "L2"

    def _build(self, ae):

        model_visible = ae._model_visible
        x_target =  ae.x_target
        reconstruction = ae.x_reconstruction_node

        dims = tf.cast(tf.reduce_prod(tf.shape(ae.raw_x)[1:]),
                       tf.float32)
        # x_shape is also a dict alternatively...
        # dims = np.prod(ae.x_shape["train"])

        # computes avg 1/2 * l2**2
        norms = tf.norm(tf.reshape(x_target - reconstruction, [-1, dims]),axis=1)
        loss = 1/2*tf.reduce_mean(tf.square(norms))
        
        return loss, [], [], []

    
