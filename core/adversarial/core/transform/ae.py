from .transform import load_network_with_dummy_x
import tensorflow as tf

def ae(dummy_x, sess, is_training, conf_file, global_step, sample_vis=False):
    ae_network = load_network_with_dummy_x(dummy_x,
                                           sess,
                                           is_training = is_training,
                                           conf_file=conf_file,
                                           global_step=global_step,
                                           base_path="vae.core")

    encoder_module = ae_network.encoder_module
    decoder_module = ae_network.decoder_module

    sample_vis_node = tf.placeholder_with_default(sample_vis, shape=(), name="sample_vis")

    # def transform(x):
    #     enc = encoder_module(x)
    #     dec = decoder_module(enc)
    #
    #     rec = tf.cond(sample_vis_node,
    #                   dec.sample,
    #                   dec.reconstruction_node
    #                   )
    #
    #     return rec

    transform = TransformAE(encoder_module, decoder_module, sample_vis_node)

    return transform, {"sample_vis" : sample_vis_node}





class TransformAE:
    def __init__(self, encoder_module, decoder_module, sample_vis_node):
        self._sample_vis_node = sample_vis_node
        self.encoder_module = encoder_module
        self.decoder_module = decoder_module


    def __call__(self, x):
        enc = self.encoder_module(x)
        self.enc = enc

        def z_mean_build():
            return self.enc

        dec = self.decoder_module(enc)

        rec = tf.cond(self._sample_vis_node,
                      dec.sample,
                      dec.reconstruction_node
                      )

        return rec
