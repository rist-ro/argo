import tensorflow as tf

from datasets.Dataset import Dataset


class AudioDataset(Dataset):
    """
    This class manage the dataset NSynth, properties of the datasets are uniquely determined
    by the params dictionary

    It compares the parameters and complete them with the default
    one. It then return a unique id identifier

    """
    default_params = {
        'crop_length': 6144,
        'shuffle_buffer': 40000,
    }

    def __init__(self, params):
        super().__init__(AudioDataset.process_params(params))

        self._crop_length_train = self._params['crop_length']
        # Width and height of each image.
        self._shuffle_buffer = self._params['shuffle_buffer']
        self._channels = 1

    def _crop_element(self, is_perturbed, crop_len, *observation):
        """
        crop element

        Args:
            is_perturbed (bool): if it is True, we expect observation = (x, x_perturbed, [y]). Where y is optional
                                if it is False, we expect observation = (x, [y]). Where y is optional
            *observation (list of tf.Tensor): a list of tensors.

        Returns:
            type: Description of returned object.

        """

        nargs = len(observation)
        y = None
        x_perturbed = None

        if is_perturbed:
            x = observation[0]
            x_perturbed = observation[1]
            if nargs == 3:
                y = observation[2]
            elif nargs > 3:
                raise Exception("observation of the dataset is a tuple with more than 3 elements.\
                                    But it is perturbed and it should be of length either 2 or 3")
        else:
            x = observation[0]
            if nargs == 2:
                y = observation[1]
            elif nargs > 2:
                raise Exception("observation of the dataset is a tuple with more than 2 elements.\
                                    But it is not perturbed and it should be of length either 1 or 2")

        x_crop = self._crop_x(x, crop_len)

        cropped_observation = (x_crop,)

        if x_perturbed is not None:
            x_perturbed_crop = self._crop_x(x_perturbed, crop_len)
            cropped_observation += (x_perturbed_crop,)

        if y is not None:
            cropped_observation += (y,)

        return cropped_observation

    def _crop_x(self, x, crop_len):
        wav = x
        # wav_sliced = tf.slice(wav, [0], [64000])

        # if signal is smaller than crop_len
        missing = tf.math.maximum(0, crop_len - tf.shape(wav)[0])
        wav = tf.cond(missing > 0, lambda: tf.pad(wav, [[0, missing], [0, 0]], 'CONSTANT'), lambda: wav)  # post padding

        crop = tf.random_crop(wav, [crop_len, self._channels])
        crop = tf.cast(crop, tf.float32)
        return crop

    @staticmethod
    def str_label_to_int(label, label_to_int_dict):
        '''
        converts the given label according to the label_to_int_dict
         Args:
             label (Tensor): scalar tensor of shape ()
             label_to_int_dict (dict): dict that specifies the mapping
        '''

        for key, value in label_to_int_dict.items():
            key = '^{}$'.format(key)  # to match whole string and not replace just a substring
            label = tf.strings.regex_replace(label, key, str(value))

        return tf.strings.to_number(label, out_type=tf.dtypes.int32)

    def int_to_str_label(self, int_label: int):
        raise NotImplementedError('Please implement this method in the desired subclass before use!')