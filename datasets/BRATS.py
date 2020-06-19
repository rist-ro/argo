"""
Module for managing BRATS dataset
"""
from datasets.BrainDataset import BrainDataset

NPROCS = 40

TRAIN_LOOP = "train_loop"
TRAIN = "train"
VALIDATION = "validation"
TEST = "test"


class BRATS(BrainDataset):
    def __init__(self, params):
        super().__init__(params)

        self._train_set_x, self._validation_set_x, self._test_set_x = self.load_float_brains(self._data_dir)

    def dataset_id(self, params):
        """
        This method interprets the parameters and generate an id
        """

        id = 'BRATS'
        id += super().dataset_id(params)

        return id

    # overriding
    @property
    def x_shape_train(self):
        return self._train_set_x_shape

    # overriding
    @property
    def x_shape_eval(self):
        return self._train_set_x_shape

    '''
    def load_slices_from_files(self, root):
        slices = []
        for path, dirs, files in os.walk(root):
            for f in files:
                fullname = os.path.abspath(os.path.join(path, f))
                slice = np.load(fullname)
                slice = slice.astype(np.float32)
                # slice = slice / (slice.max())
                slices.append(slice)
        slices = np.asarray(slices)

        return slices

    # def load_file_names(self, root):
    #     file_names = []
    #     for path, dirs, files in os.walk(root):
    #         for f in files:
    #             file_names.append(f)
    #     file_names = np.asarray(file_names)
    #     return file_names

    def load_map_filename_slice(self, root):
        map_fn_s = []
        for path, dirs, files in os.walk(root):
            for f in files:
                for slice in range(0, 130):
                    map_fn_s.append((f, slice))
        map_fn_s = np.asarray(map_fn_s)
        return map_fn_s
    
    def get_dataset_iterator(self, batch_size, dataset_str, shuffle, repeat, augment):
        is_perturbed = False
        datasets_tuple = None

        # create Dataset objects using the data previously downloaded
        if dataset_str == TRAIN:
            datasets_tuple = (self.train_set_x,)
            if hasattr(self, "perturbed_train_set_x") and self.perturbed_train_set_x is not None:
                datasets_tuple = datasets_tuple + (self.perturbed_train_set_x,)
                is_perturbed = True
            if self._train_set_y is not None:
                datasets_tuple = datasets_tuple + (self.train_set_y,)

        elif dataset_str == VALIDATION:
            datasets_tuple = (self.validation_set_x,)
            if hasattr(self, "perturbed_validation_set_x") and self.perturbed_validation_set_x is not None:
                datasets_tuple = datasets_tuple + (self.perturbed_validation_set_x,)
                is_perturbed = True
            if self._validation_set_y is not None:
                datasets_tuple = datasets_tuple + (self.validation_set_y,)

        elif dataset_str == TEST:
            datasets_tuple = (self.test_set_x,)
            if hasattr(self, "perturbed_test_set_x") and self.perturbed_test_set_x is not None:
                datasets_tuple = datasets_tuple + (self.perturbed_test_set_x,)
                is_perturbed = True
            if self._test_set_y is not None:
                datasets_tuple = datasets_tuple + (self.test_set_y,)

        else:
            raise Exception("dataset not recognized (accepted values are: train, validation and test)")

        # # CREATE TF DATASET from slices
        # # from_tensor_slices is storing dataset in the graph thus making checkpoints huge
        # dataset = tf.data.Dataset.from_tensor_slices(datasets_tuple)
        # # CREATE TF DATASET from slices

        # # CREATE TF DATASET from generator
        # def generator():
        #     for sample in zip(*datasets_tuple):
        #         yield sample
        #
        # output_types = tuple([tf.as_dtype(ds[0].dtype) for ds in datasets_tuple])
        # output_shapes = tuple([ds[0].shape for ds in datasets_tuple])
        #
        # dataset = tf.data.Dataset.from_generator(generator, output_types=output_types,
        #                                             output_shapes=output_shapes)
        # # CREATE TF DATASET from generator
        #

        # CREATE TF DATASET with map and py_func

        n_samples = datasets_tuple[0].shape[0]
        print('------------MY N SAMPLES------------', n_samples)
        dataset = tf.data.Dataset.range(n_samples)

        output_shapes = self.get_output_shapes(datasets_tuple)
        print('--------------output_shapes----------------')
        print(output_shapes)

        # why should anything be None in datasets_tuple? it is not clear that it would work with the oytput_shapes...
        # output_types = tuple([tf.as_dtype(ds[0].dtype) for ds in datasets_tuple if ds is not None])
        # output_shapes = tuple([ds[0].shape for ds in datasets_tuple if ds is not None])

        dataset = self.dataset_map(dataset, datasets_tuple)

        def _set_shapes(*nodes):
            for node, outshape in zip(nodes, output_shapes):
                node.set_shape(outshape)
            return nodes

        dataset = dataset.map(_set_shapes, num_parallel_calls=NPROCS)
        # CREATE TF DATASET with map and py_func

        # caching before shuffling and batching for super cow speed
        if self._caching_bool:
            dataset = dataset.cache()

        # PREPROCESS DATA (AUGMENT IF NEEDED)
        # if augmentation_bool:
        #     dataset = dataset.map(partial(self._preprocess_element, is_perturbed), num_parallel_calls=NPROCS)

        # SHUFFLE, REPEAT and BATCH
        # we shuffle the data and sample repeatedly batches for training
        if shuffle:
            if self._shuffling_cache is None:
                shuffling_cache = n_samples + 1
            else:
                shuffling_cache = self._shuffling_cache

            dataset = dataset.shuffle(shuffling_cache)

        if repeat:
            dataset = dataset.repeat()
            batched_dataset = dataset.batch(batch_size)
            # create iterator to retrieve batches
            iterator = batched_dataset.make_one_shot_iterator()
            # initializer = None
        else:
            batched_dataset = dataset.batch(batch_size)
            iterator = batched_dataset.make_initializable_iterator()
            # initializer = iterator.initializer

        # batched_dataset = batched_dataset.prefetch(500)

        # get a training batch of images and labels
        return iterator, is_perturbed
    '''
