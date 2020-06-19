from word_embedding.test.core.load_embeddings import get_alpha_ldv_name, load_embeddings_ldv_hdf, \
            load_emb_base, load_dict

from word_embedding.test.core.clustering.logging import instantiate_logger, CustomDefaultDict
from word_embedding.test.core.clustering.utils import get_enc, get_color_list

import os
import numpy as np

from tqdm import tqdm

from word_embedding.test.core.clustering.utils import load_csv_into_dict, get_indices, silhouette, make_frame, make_video, check_exist
from word_embedding.test.core.clustering.utils import preprocess_clusters, dist_on_sphere, do_clustering, frame_name, riemannian_normalize, riemannian_dist
from spherecluster import SphericalKMeans
from word_embedding.test.core.clustering.KMeansSim import KMeansSim
from word_embedding.test.core.clustering.RepeatedBisectionSim import RepeatedBisectionSim
from sklearn.cluster import KMeans

from functools import partial
from itertools import product
import sklearn

import time

N_INIT = 300

def create_clustering_methods(ngroups, g_matrix, n_init):
    clustering_methods = {
        # "kmsim" : (KMeansSim(n_clusters=ngroups, g_matrix=g_matrix, n_init=n_init), partial(dist_on_sphere, g_matrix=g_matrix)),
        # "krbsim" : (RepeatedBisectionSim(n_clusters=ngroups, g_matrix=g_matrix, n_init=n_init, bm='agg'), partial(dist_on_sphere, g_matrix=g_matrix)),
        "skm": (SphericalKMeans(n_clusters=ngroups, max_iter=1000, n_init=n_init), partial(dist_on_sphere, g_matrix=g_matrix)),
        # "vmfs" : (VonMisesFisherMixture(n_clusters=ngroups, n_init=n_init, posterior_type='soft'), partial(dist_on_sphere, g_matrix=g_matrix)),
        # "vmfh" : (VonMisesFisherMixture(n_clusters=ngroups, n_init=n_init, posterior_type='hard'), partial(dist_on_sphere, g_matrix=g_matrix)),
        # "km" : (KMeans(n_clusters=ngroups, n_init=n_init), partial(riemannian_dist, g_matrix=g_matrix)),
        # "lgr" : (sklearn.linear_model.LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=500), partial(dist_on_sphere, g_matrix=g_matrix)),
    }
    return clustering_methods

clustering_methods_list = list(create_clustering_methods(2, 1, 10).keys())

# groups = {
#     "animals": ["dog", "cat", "snake", "eagle", "bear", "leopard", "lion", "tiger", "wolf"],
#     "fish": ["trout", "shark", "salmon", "dolphin", "whale", "seal"],
#     "jobs": ["lawyer", "engineer", "worker", "carpenter", "musician", "doctor"],
#     "fruitsvegs": ["apple", "mushroom", "orange", "pear", "pepper"],
#     "flowers": ["orchid", "poppy", "rose", "sunflowers", "tulip"]
# }

# colors = ["brown", "cyan", "yellow", "green", "orange"]

preproc_metric_name_tuples = [
    ("nI", "I"),
    # ("std", "I"),
    # ("", "I"),
    ("nF", "F"),
    # ("std", "F"),
    # ("", "F"),
]

RUNS = 10

def create_preproc_metric_tuples(Id, F):
    norm_metric_tuples = []

    for preproc_name, metric_name in preproc_metric_name_tuples:
        if preproc_name=="nI":
            preproc = partial(riemannian_normalize, g_matrix=Id)
        elif preproc_name=="nF":
            preproc = partial(riemannian_normalize, g_matrix=F)
        elif preproc_name == "std":
            preproc = sklearn.preprocessing.scale
        elif preproc_name == "":
            preproc = None
        else:
            raise ValueError("`{:}` not recognized".format(preproc_name))

        if metric_name=="I":
            g_matrix = Id
        elif metric_name=="F":
            g_matrix = F
        else:
            raise ValueError("`{:}` not recognized".format(metric_name))

        norm_metric_tuples.append(
            (preproc_name, preproc, metric_name, g_matrix)
        )

    return norm_metric_tuples


def basic_silh_dist(preproc_name, g_matrix):
    if preproc_name.startswith("n"):
        return partial(dist_on_sphere, g_matrix=g_matrix)
    else:
        return partial(riemannian_dist, g_matrix=g_matrix)


def _tryfloat(v):
    try:
        value = float(v)
    except ValueError:
        value = v

    return value

def make_all_frames(alphas, corpus, vecsize, theta, nepoch, point, enc_name, groups_filename, baseoutdir):
    emb_dir = get_emb_dir(corpus, vecsize, nepoch)
    wemb_id = os.path.basename(os.path.normpath(emb_dir))
    _, F = load_emb_base(emb_dir, point)

    F_inv = np.linalg.inv(F)
    Id = np.eye(F.shape[0])
    preproc_metric_tuples = create_preproc_metric_tuples(Id, F)

    dictionary, idictionary, dsize = load_dict(emb_dir)
    encoder = get_enc(enc_name)

    # load groups from file
    groups = load_csv_into_dict(groups_filename, dictionary)
    groups_name = os.path.splitext(os.path.basename(groups_filename))[0]

    # make sure each group has a fixed paired colour
    colors_list = get_color_list(len(groups))
    colors = {g:c for g,c in zip(groups, colors_list)}

    output_folder = os.path.join(baseoutdir, "{:}_clusters/{:}/{:}/".format(enc_name, wemb_id, groups_name))
    os.makedirs(output_folder, exist_ok=True)

    alphas = ["limit"] + list(alphas)

    loggers = CustomDefaultDict(partial(instantiate_logger,
                                        output_folder = output_folder,
                                        log_prefix = "{:}_clustering".format(groups_name),
                                        names_to_log = ["alpha", "purity", "homogeneity", "completeness", "silhouette", "run", "time"],
                                        try_to_convert = {'alpha' : _tryfloat},
                                        field_to_sort_by = 'alpha',
                                        replace_strings_for_sort = {'limit' : -np.inf}
                                        )
                                )

    basic_loggers = CustomDefaultDict(partial(instantiate_logger,
                                        output_folder=output_folder,
                                        log_prefix="{:}_base_silhouette".format(groups_name),
                                        names_to_log=["alpha", "silhouette"],
                                        try_to_convert = {'alpha' : _tryfloat},
                                        field_to_sort_by = 'alpha',
                                        replace_strings_for_sort = {'limit' : -np.inf}
                                        )
                                )

    enum_alphas = list(enumerate(alphas))

    for preproc_name, preproc, metric_name, g_matrix in preproc_metric_tuples:
        # for norm metric
        video_folder = os.path.join(output_folder, "video_{:}".format(preproc_name))
        os.makedirs(video_folder, exist_ok=True)

        for i, alpha in tqdm(enum_alphas, desc="{:} {:} {:} {:} {:} {:} {:} {:}".format(enc_name, corpus, vecsize, nepoch, theta, point, preproc_name, metric_name)):

            ldv_name = get_alpha_ldv_name(alpha, theta + "_embeddings", point)
            ldv_path = os.path.join(emb_dir, ldv_name)

            original_norm = True if alpha != "limit" else False
            ldv = load_embeddings_ldv_hdf(ldv_path, original_norm=original_norm)
            plog = np.matmul(F_inv, np.transpose(ldv)).transpose()

            clusters = {g: plog[get_indices(dictionary, words)] for g, words in groups.items()}

            # for norm, metric
            points_prep, len_list, group_keys, labels = preprocess_clusters(clusters, preproc_method=preproc, repack=False)
            perm = np.random.permutation(points_prep.shape[0])
            shuffled_points = points_prep[perm]
            shuffled_labels = labels[perm]

            make_frame(video_folder, encoder, clusters, theta, point, enc_name, preproc_name, alpha, i, colors)

            baselog = basic_loggers[(wemb_id, theta, point, preproc_name, metric_name)]
            if not baselog.has_values(["alpha"], [alpha]):
                silh = silhouette(shuffled_points, shuffled_labels, basic_silh_dist(preproc_name, g_matrix))
                baselog.log([alpha, silh])

            ngroups = len(group_keys)
            clustering_methods = create_clustering_methods(ngroups, g_matrix, n_init=N_INIT)

            for cname, (cobj, cdist) in clustering_methods.items():
                # skip redundant clusterings
                if cname in ["kmsim", "krbsim"] and preproc_name=="":
                    continue

                logger = loggers[(wemb_id, theta, point, preproc_name, metric_name, cname)]
                for run in range(RUNS):
                    if not logger.has_values(["run", "alpha"], [run, alpha]):
                        ti = time.time()
                        pur, hom, comp, silh = do_clustering(cobj, shuffled_points, shuffled_labels, cdist)
                        tf = time.time()
                        logger.log([alpha, pur, hom, comp, silh, run, tf-ti])


    for preproc_name, preproc, metric_name, g_matrix in preproc_metric_tuples:
        # for norm, metric
        baselog = basic_loggers[(wemb_id, theta, point, preproc_name, metric_name)]
        baselog.plot(x="alpha", y="silhouette", ylim=(-1, 1))
        for cname in clustering_methods_list:
            logger = loggers[(wemb_id, theta, point, preproc_name, metric_name, cname)]
            logger.plot_errorbar(x="alpha", y="purity", ylim=(0, 1), string_replace=replace_strings_for_plots, suffix="-pur")
            logger.plot_errorbar(x="alpha", y="silhouette", ylim=(-1, 1), string_replace=replace_strings_for_plots, suffix="-silh")

        video_folder = os.path.join(output_folder, "video_{:}".format(preproc_name))
        make_video(video_folder, frame_name(theta, point, enc_name, preproc_name))


base_emb_dir = "/data1/text/alpha-embeddings/"

def get_emb_dir(corpus, vecsize, nepoch):
    return os.path.join(base_emb_dir, corpus+"-alpha-emb", "{:}-v{:}-n{:}".format(corpus,vecsize,nepoch))


groups_dirname = "/data/captamerica_hd2/text/concept_categorization_datasets/"
groups_names = ["ap", "bless"] # "battig", "capitals", , "essli-2008"

groups_filenames = [os.path.join(groups_dirname, gn+".csv") for gn in groups_names]

baseoutdir = '/data2/text/clustering_norm'

corpora = ["geb"] #"enwiki"]
thetas = ["u+v"] #"u"
vecsizes = [300] #, 50, 2]
nepochs = [1000] #, 100]
point_names = ["0", "u", "ud"]

base_dir = {
    "geb" : "/data/wonderwoman_hd2/text/word_embeddings",
    "enwiki": "/data/thor_hd2/text/word_embeddings",
}

DIGITS_ALPHA = 2
array_of_ranges = [
    np.arange(-2, 2.1, 0.1),
    np.arange(-5, 5, 1.)
]

alphas = np.sort(list(set(map(partial(round, ndigits=DIGITS_ALPHA),
                            np.concatenate(array_of_ranges)))))

# solve the ambiguity, 0, -0
alphas = np.array([a if not np.isclose(a, 0.) else 0. for a in alphas])


replace_strings_for_plots = {
    'limit' : alphas.min()-1., # since for plotting I need to choose a value
}


for gfn, corpus, vecsize, theta, nepoch, point in product(groups_filenames, corpora, vecsizes, thetas, nepochs, point_names):
    make_all_frames(alphas, corpus, vecsize, theta, nepoch, point, "pca", gfn, baseoutdir)
