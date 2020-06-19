import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf
import os
from .Dataset import Dataset, TRAIN, VALIDATION, TEST

os.environ['NUMEXPR_MAX_THREADS'] = '20'

from word_embedding.test.core.load_embeddings import load_embeddings_ldv_hdf, load_dict, load_emb_base, get_embpath
from word_embedding.test.core.measures import center_and_normalize_riemannian
from word_embedding.preprocess.preprocess_lib import preproc_a

NPROCS = 4

class IMDBReviews(Dataset):

    default_params = {
        'emb_dir': "/data1/text/new-alpha-embeddings/geb-alpha-emb/geb-v50-n1000",
        'alpha' : 1., # can also be "limit"
        'theta': "u+v",
        'point': "0",
        'norm' : "F",
        'aggregate' : False,
        'dataset_cache' : False
    }

    def __init__(self, params):
        super().__init__(IMDBReviews.process_params(params))

        self._id = self.dataset_id(self._params)
        emb_dir = self._params["emb_dir"]
        alpha_tuple = self._params["alpha"]
        theta = self._params["theta"]
        point = self._params["point"]
        normalization = self._params["norm"]
        self._aggregate = self._params["aggregate"]

        self._emb_id = os.path.basename(os.path.normpath(emb_dir))
        # corpus, vstr, nstr = emb_id.split('-')

        vocab, ivocab, vocab_size = load_dict(emb_dir)
        self._dictionary = vocab

        # ALPHA BASE
        alphas, I = load_emb_base(emb_dir, point)
        I_inv = np.linalg.inv(I)

        embpath = get_embpath(alpha_tuple, theta, point, emb_dir)

        # fullname = os.path.join(emb_dir, get_alpha_ldv_name(alpha, theta+"_embeddings", point))
        ldv = load_embeddings_ldv_hdf(embpath)

        plog = np.matmul(I_inv, np.transpose(ldv)).transpose()
        self._emb_size = plog.shape[1]

        if normalization is not None:
            if normalization=="I":
                norm_matrix = np.eye(self._emb_size)
            elif normalization=="F":
                norm_matrix = I
            else:
                raise Exception(
                    "Only Identity (I) or Fisher (F) normalization are allowed.. or no normalization (None)")

            plog = center_and_normalize_riemannian(plog, norm_matrix, center=False)

        unk_emb = np.expand_dims(np.zeros(plog.shape[1]), axis=0)

        self._embeddings = np.array(np.concatenate((plog, unk_emb), axis=0), dtype=np.float32)

        train_dataset, info = tfds.load('imdb_reviews/plain_text', split="train", as_supervised=True, with_info=True)
        self._n_samples_train = info.splits['train'].num_examples
        # val/test equal split
        self._n_samples_validation = info.splits['test'].num_examples/2.
        self._n_samples_test = info.splits['test'].num_examples/2.

        # # new S3 API, still not supported by imdb reviews
        # validation_dataset = tfds.load('imdb_reviews/plain_text', split="test[:50%]", as_supervised=True)
        # test_dataset = tfds.load('imdb_reviews/plain_text', split="test[-50%:]", as_supervised=True)

        # legacy API
        validation_split, test_split = tfds.Split.TEST.subsplit(k=2)
        validation_dataset = tfds.load('imdb_reviews/plain_text', split=validation_split, as_supervised=True)
        test_dataset = tfds.load('imdb_reviews/plain_text', split=test_split, as_supervised=True)

        self._raw_datasets = {
            TRAIN : train_dataset,
            VALIDATION : validation_dataset,
            TEST : test_dataset
        }

        self._shuffling_cache = None

        self._dataset_cache = self._params["dataset_cache"]

        if self._aggregate:
            self._x_sample_shape = (self._emb_size, )
        else:
            self._x_sample_shape = (None, self._emb_size)

        # binary classification (I treat it as softmax for simplicity), the output must be 2
        self._y_sample_shape = ()
        self._y_one_hot_sample_shape = (2,)

        self._padded_shape = (self._x_sample_shape, self._y_sample_shape)

    def dataset_id(self, params):
        """
        This method interprets the parameters and generate an id
        """

        IMDBReviews.check_params_impl(params)

        _id = 'IMDBReviews'
        emb_dir = params["emb_dir"]
        alpha_tuple = params["alpha"]
        theta = params["theta"]
        point = params["point"]
        emb_id = os.path.basename(os.path.normpath(emb_dir))
        _id += "-" + emb_id

        alpha_id, alpha_kwargs = alpha_tuple
        _id+= "-a"
        if alpha_id=='float':
            _id += "float_v{:.2f}".format(alpha_kwargs['value'])
        elif alpha_id == 'limit':
            _id += "limit"
            _id += "_t{:d}".format(alpha_kwargs['ntop'])
            _id += "_w{:d}".format(int(bool(alpha_kwargs['weighted'])))
            _id += "_f{:d}".format(int(bool(alpha_kwargs['fraction'])))
        else:
            raise ValueError("Option `{:}`not recognized for alpha_id. Only `float` or `limit` are allowed.".format(alpha_id))
       
       #try:
       #     _id += "-a{:.2f}".format(alpha)
       # except:
       #     _id += "-a{:}".format(alpha)

        _id += "-" + theta
        _id += "-" + point

        normalization = params["norm"]
        if normalization is not None:
            if normalization not in ["I", "F"]:
                raise Exception("Only Identity (I) or Fisher (F) normalization are allowed..")
            _id += "-norm" + normalization

        aggregate = params["aggregate"]

        if aggregate:
            _id += "-aggr"

        return _id

    @property
    def n_samples_train(self):
        return self._n_samples_train

    @property
    def x_shape_train(self):
        """return the shape of an input sample"""
        return self._x_sample_shape

    @property
    def x_shape_eval(self):
        """return the shape of an input sample"""
        return self._x_sample_shape

    @property
    def y_shape(self):
        """return the shape of an output sample"""
        return self._y_one_hot_sample_shape

    @property
    def n_labels(self):
        """return the number of labeles in this dataset"""
        return 2

    @property
    def labels(self):
        """return the list of labels in this dataset"""
        return [0, 1]

    def _preprocess(self, excerpt, label):
        indexes = np.array([self._dictionary[w] if w in self._dictionary else -1 for w in preproc_a(excerpt)])

        xdata = self._embeddings[indexes]
        if self._aggregate:
            xdata = np.mean(xdata, axis=0)

        return xdata, np.array(label, dtype=np.int32)

    def get_dataset_iterator(self, batch_size, dataset_str, shuffle, repeat, augment):

        dataset = self._raw_datasets[dataset_str]

        output_types = (tf.float32, tf.int32
                        
                    )
        dataset = dataset.map(
            lambda x ,y: tuple(tf.py_func(self._preprocess,
                                         [x ,y], output_types)
                              ), num_parallel_calls=NPROCS)

        output_shapes = (self._x_sample_shape, self._y_sample_shape)

        def _set_shapes(*nodes):
            for node, outshape in zip(nodes, output_shapes):
                node.set_shape(outshape)
            return nodes

        dataset = dataset.map(_set_shapes, num_parallel_calls=NPROCS)
        if self._dataset_cache:
            dataset = dataset.cache()

        if shuffle:
            if self._shuffling_cache is None:
                shuffling_cache = self._n_samples_train + 1
            else:
                shuffling_cache = self._shuffling_cache

            dataset = dataset.shuffle(shuffling_cache)

        if repeat:
            dataset = dataset.repeat()

        batched_dataset = dataset.padded_batch(batch_size, padded_shapes=self._padded_shape)

        if repeat:
            # create iterator to retrieve batches
            iterator = batched_dataset.make_one_shot_iterator()
        else:
            iterator = batched_dataset.make_initializable_iterator()

        return iterator, False


