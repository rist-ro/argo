import numpy as np
import os
from .Dataset import Dataset


os.environ['NUMEXPR_MAX_THREADS'] = '20'

from word_embedding.test.core.load_embeddings import load_embeddings_ldv_hdf, load_dict, load_emb_base, get_embpath
from word_embedding.test.core.measures import center_and_normalize_riemannian
from word_embedding.preprocess.preprocess_lib import preproc_a

import multiprocessing

NPROCS = 4

class AlphaDatasetArr(Dataset):

    default_params = {
        'alpha' : ('float', {'value' : 1.}), # ('limit', {'ntop' : 1, 'weighted' : False, 'fraction' : True})
        'theta': "u+v",
        'point': "0",
        'norm' : "F",
        'aggregate' : True,
    }

    def __init__(self, params):
        super().__init__(AlphaDatasetArr.process_params(params))

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

    def _set_shapes(self, n_samples_train, n_labels):

        self._n_labels = n_labels
        self._n_samples_train = n_samples_train

        if self._aggregate:
            self._x_sample_shape = (self._emb_size,)
        else:
            self._x_sample_shape = (None, self._emb_size)

        # binary classification (I treat it as softmax for simplicity), the output must be 2
        self._y_sample_shape = ()
        self._y_one_hot_sample_shape = (self._n_labels,)

        self._padded_shape = (self._x_sample_shape, self._y_sample_shape)


    def alpha_id_str(self, alpha_tuple):
        alpha_id, alpha_kwargs = alpha_tuple
        _id = '-a'

        if alpha_id=='float':
            _id += "float_v{:.2f}".format(alpha_kwargs['value'])
        elif alpha_id == 'limit':
            _id += "limit"
            _id += "_t{:d}".format(alpha_kwargs['ntop'])
            _id += "_w{:d}".format(int(bool(alpha_kwargs['weighted'])))
            _id += "_f{:d}".format(int(bool(alpha_kwargs['fraction'])))
        else:
            raise ValueError("Option `{:}`not recognized for alpha_id. Only `float` or `limit` are allowed.".format(alpha_id))

        return _id

    def dataset_id(self, params):
        """
        This method interprets the parameters and generate an id
        """

        AlphaDatasetArr.check_params_impl(params)

        emb_dir = params["emb_dir"]
        alpha_tuple = params["alpha"]
        theta = params["theta"]
        point = params["point"]
        emb_id = os.path.basename(os.path.normpath(emb_dir))
        _id = "-" + emb_id

        _id += self.alpha_id_str(alpha_tuple)

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
        """return the number of labels in this dataset"""
        return self._n_labels

    @property
    def labels(self):
        """return the list of labels in this dataset"""
        return np.arange(self._n_labels)

    def _preprocess(self, excerpt, label):
        indexes = np.array([self._dictionary[w] if w in self._dictionary else -1 for w in preproc_a(excerpt)])
        if len(indexes)==0:
            return None, None

        try:
            xdata = self._embeddings[indexes]
        except:
            raise Exception("could not parse excerpt: `{:}`, \n\n indexes found were: `{:}`".format(excerpt, indexes))

        if self._aggregate:
            xdata = np.mean(xdata, axis=0)

        return xdata, np.array(label, dtype=np.int32)

    def _preprocess_arrays(self, data, target):
        # pool = multiprocessing.Pool(processes=NPROCS)

        preprocessed_tuples = list(map(self._preprocess, data, target))

        preprocessed_tuples = [(x,y) for x,y in preprocessed_tuples if x is not None]

        prep_data, prep_target = zip(*preprocessed_tuples)
        prep_data = np.array(prep_data)
        prep_target = np.array(prep_target)
        return prep_data, prep_target
