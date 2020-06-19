from .transform import load_network_with_dummy_x
from tensorflow_probability import distributions as tfd
import tensorflow as tf

def vae(dummy_x, sess, is_training, conf_file, global_step=None, sample_hid=False, sample_vis=False, z_std_scale=1.):
    vae_network = load_network_with_dummy_x(dummy_x,
                                            sess,
                                            is_training=is_training,
                                            conf_file=conf_file,
                                            global_step=global_step,
                                            base_path="vae.core")

    encoder_module = vae_network.encoder_module
    decoder_module = vae_network.decoder_module

    sample_hid_node = tf.placeholder_with_default(sample_hid, shape=(), name="sample_hid")
    sample_vis_node = tf.placeholder_with_default(sample_vis, shape=(), name="sample_vis")
    z_std_scale_node = tf.placeholder_with_default(z_std_scale, shape=(), name="z_std_scale")


    # def transform(x):
    #     enc = encoder_module(x)
    #
    #     def sample_z_build():
    #         mean = enc.mean()
    #         std = enc.scale
    #         return tfd.MultivariateNormalDiag(loc=mean, scale_diag=std * z_std_scale_node).sample()
    #
    #     def z_mean_build():
    #         return enc.mean()
    #
    #     z = tf.cond(sample_hid_node,
    #                 sample_z_build,
    #                 z_mean_build)
    #
    #     dec = decoder_module(z)
    #
    #     rec = tf.cond(sample_vis_node,
    #                   dec.sample,
    #                   dec.reconstruction_node
    #                   )
    #
    #     return rec

    transform = TransformVAE(encoder_module, decoder_module, sample_hid_node, sample_vis_node, z_std_scale_node)

    return transform, {"sample_hid": sample_hid_node,
                       "sample_vis" : sample_vis_node,
                       "z_std_scale" : z_std_scale_node}


class TransformVAE:
    def __init__(self, encoder_module, decoder_module, sample_hid_node, sample_vis_node, z_std_scale_node):
        self._sample_hid_node = sample_hid_node
        self._sample_vis_node = sample_vis_node
        self._z_std_scale_node = z_std_scale_node
        self.encoder_module = encoder_module
        self.decoder_module = decoder_module


    def __call__(self, x):
        #import pdb;pdb.set_trace()
        enc = self.encoder_module(x)
        self.enc = enc

        def sample_z_build():
            mean = enc.mean()
            std = enc.scale
            return tfd.MultivariateNormalDiag(loc=mean, scale_diag=std * self._z_std_scale_node).sample()

        def z_mean_build():
            return enc.mean()

        z = tf.cond(self._sample_hid_node,
                    sample_z_build,
                    z_mean_build)

        dec = self.decoder_module(z)

        rec = tf.cond(self._sample_vis_node,
                      dec.sample,
                      dec.reconstruction_node
                      )

        return rec
