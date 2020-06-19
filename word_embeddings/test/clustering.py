from word_embedding.test.core.load_embeddings import load_pretrained_glove, load_pretrained_w2v, load_glove, load_w2v, info_pretrained
import numpy as np
import sklearn
import os
from sklearn.cluster import KMeans
from functools import partial

from word_embedding.test.core.clustering.KMeansSim import KMeansSim
from word_embedding.test.core.clustering.RepeatedBisectionSim import RepeatedBisectionSim
from word_embedding.test.core.clustering.logging import instantiate_logger, CustomDefaultDict

from word_embedding.test.core.clustering.utils import load_csv_into_dict, get_indices, silhouette, print_scores, compute_scores
from word_embedding.test.core.clustering.utils import cos_similarity, dist_on_sphere, riemannian_dist, make_clusters_out_of, do_clustering
from spherecluster import SphericalKMeans
from spherecluster import VonMisesFisherMixture
from tqdm import tqdm

groups_dirname = "/data/captamerica_hd2/text/concept_categorization_datasets/"
groups_names = ["ap", "battig", "capitals", "bless", "essli-2008"]

groups_filenames = [os.path.join(groups_dirname, gn+".csv") for gn in groups_names]

w2vdir = "/data2/text/word_embeddings/word2vec"
# references: "wikigiga5", "commoncrawl840B", "googlenews"
references = {
    "wikigiga5" : (load_pretrained_glove, {"gname" : "wikigiga5"}, ["u"]),
    "commoncrawl840B": (load_pretrained_glove, {"gname" : "commoncrawl840B"}, ["u"]),
    "googlenews": (load_pretrained_w2v, {"wname" : "googlenews"}, ["u"]),
    "w2v"  : (load_w2v, [
                        {"base_dir": w2vdir, "corpus" : "geb",
                              "vecsize" : 300, "nepoch": 50,
                              "sw" : 10, "ns" : 10},
                        {"base_dir": w2vdir, "corpus" : "geb",
                              "vecsize" : 300, "nepoch": 50,
                              "sw" : 5, "ns" : 10},
                        {"base_dir": w2vdir, "corpus" : "geb",
                              "vecsize" : 400, "nepoch": 50,
                              "sw" : 5, "ns" : 10}
                        ],
              ["u", "u+v"]
              ),

}

def get_ref_vecsize(name, kwargs):
    try:
        vecsize = kwargs["vecsize"]
    except:
        vecsize = info_pretrained[name][1]
    return vecsize

def get_ref_name(name, kwargs):
    if name in info_pretrained:
        return name

    return "{:}-{:}-v{:}-n{:}-sw{:}-ns{:}".format(name, kwargs["corpus"],
                                                  kwargs["vecsize"], kwargs["nepoch"],
                                                  kwargs["sw"], kwargs["ns"])

def get_ref_theta_name(name, thname):

    if name in info_pretrained:
        tns = thname.split("_")
        if len(tns)==1:
            newthname = ""
        else:
            newthname = tns[1]

        return newthname

    else:
        return thname

# mine:
corpora = ["geb"], # "enwiki"]
thetas = ["u", "u+v"]
vecsizes = [2, 50, 300]
nepochs = [100, 1000], #200, 300, 400, 500, 600, 700, 800, 900, 1000]
base_dir = {
    "geb" : "/data/wonderwoman_hd2/text/word_embeddings",
    "enwiki": "/data/thor_hd2/text/word_embeddings",
}

output_folder = "/data2/text/clustering/"

def basic_silh_dist(thname, g_matrix):
    return partial(riemannian_dist, g_matrix=g_matrix)

def create_clustering_methods(ngroups, g_matrix, n_init):
    clustering_methods = {
        "kmsim" : (KMeansSim(n_clusters=ngroups, g_matrix=g_matrix, n_init=n_init), partial(dist_on_sphere, g_matrix=g_matrix)),
        "krbsim" : (RepeatedBisectionSim(n_clusters=ngroups, g_matrix=g_matrix, n_init=n_init, bm='agg'), partial(dist_on_sphere, g_matrix=g_matrix)),
        "skm": (SphericalKMeans(n_clusters=ngroups, n_init=n_init), partial(dist_on_sphere, g_matrix=g_matrix)),
        # "vmfs" : (VonMisesFisherMixture(n_clusters=ngroups, n_init=n_init, posterior_type='soft'), partial(dist_on_sphere, g_matrix=g_matrix)),
        # "vmfh" : (VonMisesFisherMixture(n_clusters=ngroups, n_init=n_init, posterior_type='hard'), partial(dist_on_sphere, g_matrix=g_matrix)),
        "km" : (KMeans(n_clusters=ngroups, n_init=n_init), partial(dist_on_sphere, g_matrix=g_matrix)),
        "lgr" : (sklearn.linear_model.LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=500), partial(dist_on_sphere, g_matrix=g_matrix)),
    }
    return clustering_methods

# clustering methods which are possible to apply also to not normalized or standardized data
clmethods_for_not_preproc = ["km", "lgr"]



N_INIT = 1000
NORMTOL = 1e-15

def cluster_references(groups_filename, output_folder, references):
    groups_name = os.path.splitext(os.path.basename(groups_filename))[0]

    loggers = CustomDefaultDict(partial(instantiate_logger,
                                        output_folder=output_folder,
                                        log_prefix="{:}_clustering".format(groups_name),
                                        names_to_log=["purity", "homogeneity", "completeness", "silhouette"]
                                        )
                                )

    basic_loggers = CustomDefaultDict(partial(instantiate_logger,
                                        output_folder=output_folder,
                                        log_prefix="{:}_base_silhouette".format(groups_name),
                                        names_to_log=["silhouette"]
                                        )
                                )

    for refname, (refloader, refloadkwargs_list, refthetas) in references.items():
        refloadkwargs_list = refloadkwargs_list if isinstance(refloadkwargs_list, list) else [refloadkwargs_list]

        for refloadkwargs in refloadkwargs_list:
            outbasename = get_ref_name(refname, refloadkwargs)
            vecsize = get_ref_vecsize(refname, refloadkwargs)
            g_matrix = np.eye(vecsize)

            dictionary, vecs = refloader(**refloadkwargs)

            groups = load_csv_into_dict(groups_filename, dictionary)
            ngroups = len(groups)

            clustering_methods = create_clustering_methods(ngroups, g_matrix, n_init=N_INIT)

            for th in refthetas:
                points_dict = prepare_clusters(vecs, th, groups, dictionary)

                for thname, (points, labels) in tqdm(points_dict.items(), desc=refname+"-"+th):
                    #in the reference case, we don't want to write "u" since it is depending on the reference (usually "u" but loads "u+v", hence the confusion)
                    thname = get_ref_theta_name(refname, thname)

                    baselog = basic_loggers[(outbasename, thname)]
                    success, _ = baselog.try_read_csv()
                    if not success:
                        silh = silhouette(points, labels, basic_silh_dist(thname, g_matrix))
                        baselog.log([silh])

                    for cname, (cobj, cdist) in clustering_methods.items():
                        logger = loggers[(outbasename, thname, cname)]
                        success, _ = logger.try_read_csv()
                        if not success:
                            pur, hom, comp, silh = do_clustering(cobj, points, labels, cdist)
                            logger.log([pur, hom, comp, silh])


def cluster_glove_trained(groups_filename, output_folder, corpora, thetas, vecsizes, nepochs):
    groups_name = os.path.splitext(os.path.basename(groups_filename))[0]

    loggers = CustomDefaultDict(partial(instantiate_logger,
                                        output_folder=output_folder,
                                        log_prefix="{:}_clustering_glove".format(groups_name),
                                        names_to_log=["epoch", "purity", "homogeneity", "completeness", "silhouette"]
                                        )
                                )

    basic_loggers = CustomDefaultDict(partial(instantiate_logger,
                                              output_folder=output_folder,
                                              log_prefix="{:}_base_silhouette_glove".format(groups_name),
                                              names_to_log=["epoch", "silhouette"]
                                              )
                                      )

    for vecsize in vecsizes:
        g_matrix = np.eye(vecsize)

        for corpus in corpora:
            wename = "{:}_v{:}".format(corpus, vecsize)
            for nepoch in tqdm(nepochs, desc=wename):

                dictionary, vecs, _ = load_glove(corpus, vecsize, nepoch, calc_prob=False,
                                                 glove_dir=base_dir[corpus])

                groups = load_csv_into_dict(groups_filename, dictionary)
                ngroups = len(groups)
                clustering_methods = create_clustering_methods(ngroups, g_matrix, n_init=N_INIT)

                for theta in thetas:
                    points_dict = prepare_clusters(vecs, theta, groups, dictionary)

                    for thname, (points, labels) in points_dict.items():
                        baselog = basic_loggers[(wename, thname)]
                        if not baselog.has_value("epoch", nepoch):
                            silh = silhouette(points, labels, basic_silh_dist(thname, g_matrix))
                            baselog.log([nepoch, silh])

                        for cname, (cobj, cdist) in clustering_methods.items():
                            # skip if the method is not meaningful
                            if (cname not in clmethods_for_not_preproc) and is_preproc(thname):
                                continue

                            logger = loggers[(wename, thname, cname)]
                            if not logger.has_value("epoch", nepoch):
                                pur, hom, comp, silh = do_clustering(cobj, points, labels, cdist)
                                logger.log([nepoch, pur, hom, comp, silh])

            # for theta in thetas:
            #     for thname in [theta, theta+"_norm", theta+"_std"]:
            #         for cname in clustering_methods:
            #             key_tuple = (wename, thname, cname)
            #             loggers[key_tuple].plot(x="epoch")
            for nametuple, logger in loggers.items():
                logger.plot(x="epoch")


def is_preproc(thname):
    return ("norm" in thname or "std" in thname)


def get_embeddings(vecs, theta):
    if theta=="u":
        return vecs["u"]
    elif theta=="u+v":
        return  vecs["u"]+vecs["v"]
    else:
        raise ValueError("theta option not recognized")


def points_and_labels_from_clusters(clusters):
    points_list, labels_list = zip(*[(clusters[g], [i] * clusters[g].shape[0]) for i, g in enumerate(clusters.keys())])

    clusters_points = np.concatenate(points_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    perm = np.random.permutation(clusters_points.shape[0])
    shuffled_clusters_points = clusters_points[perm]
    shuffled_labels = labels[perm]

    return clusters_points, labels, shuffled_clusters_points, shuffled_labels


def prepare_clusters(vecs, theta, groups, dictionary):
    embeddings = get_embeddings(vecs, theta)
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1).reshape(-1, 1) + NORMTOL)
    embeddings_std = sklearn.preprocessing.scale(embeddings)

    clusters, clusters_norm, clusters_std = make_clusters_out_of(
        [embeddings, embeddings_norm, embeddings_std], groups, dictionary)

    _, _, points, labels = points_and_labels_from_clusters(clusters)
    _, _, points_norm, labels_norm = points_and_labels_from_clusters(clusters_norm)
    _, _, points_std, labels_std = points_and_labels_from_clusters(clusters_std)

    return {theta : (points, labels),
            theta+"_norm" : (points_norm, labels_norm),
            theta+"_std" : (points_std, labels_std),
            }


def read_groups(groups_filename, dictionary):
    groups_name = os.path.splitext(os.path.basename(groups_filename))[0]
    groups = load_csv_into_dict(groups_filename, dictionary)
    return groups, groups_name


if __name__ == "__main__":
    for gfn in groups_filenames:
        cluster_references(gfn, output_folder, references)
        cluster_glove_trained(gfn, output_folder, corpora, thetas, vecsizes, nepochs)