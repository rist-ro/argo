from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as MTSNE
import pandas as pd
import numpy as np
import sklearn
import os

from scipy.linalg import sqrtm
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from itertools import islice
import subprocess
from word_embedding.test.core.plotting import initialize_plot, finalize_plot
from word_embedding.test.core.spaces import dist_sphere_riem_amb, mean_on_sphere


def load_csv_into_dict(filename, dictionary):
    df = pd.read_csv(filename, sep=" ")
    tot_words = len(df)
    df = df[df['word'].isin(dictionary)]
    in_dict = len(df)
    print("words in_dict/total: %.2f%% (%d/%d)" %(100 * in_dict / float(tot_words), in_dict, tot_words))
    groups = df.groupby('category')['word'].apply(list).to_dict()

    return groups

def frame_path(video_folder, t, p, enc, m, i):
    return os.path.join(video_folder, frame_name(t,p,enc,m)+'-{:04d}.png'.format(i))

def frame_name(t, p, enc, m):
    return '{:}_emb-{:}-{:}_{:}'.format(t, p, enc, m)

# to make video from bash
# ffmpeg -framerate 2 -i u_emb-0-%04d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p first100-u_emb-0-pca.mp4
def make_video(folder, name):
    command = ['ffmpeg', '-y', '-framerate', '2', '-i', name+'-%04d.png',
               '-c:v', 'libx264', '-profile:v', 'high', '-crf', '20', '-pix_fmt', 'yuv420p',
               '-vf', 'scale=640:-2', name+'.mp4']
    p = subprocess.Popen(command, stderr=subprocess.DEVNULL, cwd=folder)
    p.wait()

def make_frame(folder, encoder, clusters_dict, theta, point, enc_name, norm, alpha, i, colors):
    img_path = frame_path(folder, theta, point, enc_name, norm, i)
    if check_exist(img_path):
        return

    cluster_points_2d = low_dim_clust(encoder, clusters_dict)

    extreme = 1.01
    scatter_plot(cluster_points_2d, img_path, alpha, extreme, colors)


def get_indices(dictionary, words_list):
    return np.array([dictionary[w] for w in words_list])


N_COMPONENTS = 2
def get_enc(enc_name):
    if enc_name == "pca":
        encoder = PCA(n_components=N_COMPONENTS)
    elif enc_name == "tsne":
        encoder = MTSNE(n_components=N_COMPONENTS, n_jobs=40)
    else:
        raise Exception("option not recognized: {:}".format(enc_name))

    return encoder

def get_color_list(n):
    if n<=10:
        colors = list(mcolors.TABLEAU_COLORS.values())
    elif n>10 and n<=20:
        colors = list(matplotlib.cm.get_cmap("tab20").colors)
    else:
        cmap = matplotlib.cm.get_cmap("gist_rainbow", n)
        colors = [cmap(i) for i in range(n)]

    return [mcolors.to_hex(rgba) for rgba in colors]


def do_clustering(cluster_obj, points, labels, dist_method):
    # try to fit unsupervised, otherwise fit supervised
    try:
        cluster_obj.fit(points)
        predictions = cluster_obj.labels_
    except TypeError:
        cluster_obj.fit(points, labels)
        predictions = cluster_obj.predict(points)

    pur, hom, comp, silh = compute_scores(labels, predictions, points, dist_method)
    return pur, hom, comp, silh

def low_dim_clusters(encoder, clusters_list):
    clusters_2d_list = []

    for clusters in clusters_list:
        clusters_2d = low_dim_clust(encoder, clusters)
        clusters_2d_list.append(clusters_2d)

    return clusters_2d_list

def low_dim_clust(encoder, clusters):
    points, len_list, group_keys, labels = unpack_clusters_dict(clusters)

    # for viz
    points_2d = encoder.fit_transform(points)
    #points_2d = points_2d / np.max(points_2d)
    points_2d = sklearn.preprocessing.scale(points_2d)/3.

    clusters_2d = repack_clusters_dict(points_2d, len_list, group_keys)

    return clusters_2d

def low_dim_cluster_points(encoder, points):
    # for viz
    points_2d = encoder.fit_transform(points)
    points_2d = points_2d / np.max(points_2d)  # sklearn.preprocessing.scale(points_2d)/2.

    return points_2d

def unpack_clusters_dict(clusters):
    group_keys = list(clusters.keys())
    points_list, len_list = zip(*[(clusters[g], clusters[g].shape[0]) for g in group_keys])

    points = np.concatenate(points_list, axis=0)

    labels_list = [[i] * size for i, size in enumerate(len_list)]
    labels = np.concatenate(labels_list, axis=0)

    return points, len_list, group_keys, labels

def repack_clusters_dict(points, len_list, group_keys):
    it = iter(list(points))
    points_cl_list = [list(islice(it, size)) for size in len_list]

    clusters = {g: points_cl_list[i] for i, g in enumerate(group_keys)}

    return clusters

def make_clusters_out_of(embeddings_list, groups, dictionary):
    clusters_list = []
    for embeddings in embeddings_list:
        clusters = {g: embeddings[get_indices(dictionary, words)] for g, words in groups.items()}
        clusters_list.append(clusters)

    return clusters_list

def preprocess_clusters(clusters, preproc_method=None, repack=False):

    points, len_list, group_keys, labels = unpack_clusters_dict(clusters)

    if preproc_method is not None:
        points = preproc_method(points)

    if repack:
        preproc_clusters = repack_clusters_dict(points, len_list, group_keys)
        return preproc_clusters
    else:
        return points, len_list, group_keys, labels


def expected_keys(kwargs, n, keys):
    assert len(kwargs) == n, "expected len `{:}` for kwargs dict `{:}`.".format(n, kwargs)

    for k in keys:
        if not hasattr(kwargs, k):
            raise Exception("expected keys `{:}` for kwargs dict `{:}`. `{:}` not found.".format(keys, kwargs, k))


def check_exist(path):
    return os.path.exists(path) and os.path.getsize(path) > 0

def scatter_plot(clusters_2d, img_path, alpha, extreme, colors):
    alpha_str = alpha if alpha == "limit" else "{:.2f}".format(alpha)
    axes = initialize_plot(title="alpha = {:}".format(alpha_str)) # , figsize=[6.4*2, 4.8*2])
    for g, points in clusters_2d.items():
        x, y = zip(*points)
        plt.scatter(x, y, c=colors[g], label=g)
    xlim = (-extreme, extreme)
    ylim = (-extreme, extreme)
    finalize_plot(axes, img_path, xlim, ylim, legendloc="outside")
    plt.close()


def mean_centroid_dist(points, g_matrix, space="flat"):
    if space=="flat":
        # centroid is same with any metric, derive mean square error
        centroid = np.mean(points, axis=0)
        # I want a row vector
        centroid = centroid.reshape(1,-1)

        # this is one row vector
        IC = np.matmul(g_matrix, centroid.reshape(-1, 1)).T

        sqnormc = np.sum(centroid * IC, axis=1).reshape(-1)

        npoints = points.shape[0]
        dot_part = -2 * np.matmul(points, IC.T).reshape(-1)
        ICp = np.matmul(g_matrix, points.T).T
        sqnormp = np.sum(points * ICp, axis=1).reshape(-1)
        np.testing.assert_array_equal(list(map(len, [sqnormc, sqnormp, dot_part])), [1, npoints, npoints])
        sqdists = sqnormp + sqnormc + dot_part

        sqdists = check_sqdist_numerics(sqdists)
        dists = np.sqrt(sqdists)

    elif space == "sphere":
        centroid = mean_on_sphere(points)
        centroid = centroid.reshape(1, -1)
        dists = dist_sphere_riem_amb(points, centroid, g_matrix)

    else:
        raise ValueError("space option not recognized")

    return np.mean(dists)

def silhouette(clusters_points, predictions, dist_method):
    distances = dist_method(clusters_points, clusters_points)
    silh = sklearn.metrics.silhouette_score(distances, predictions, metric="precomputed")
    return silh

def _silhouette(clusters, g_matrix, space="flat"):
    points_list, labels_list = zip(*[(clusters[g], [i]*clusters[g].shape[0]) for i,g in enumerate(clusters.keys())])

    points = np.concatenate(points_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    xp = points

    if space=="flat":
        Ixp = np.matmul(g_matrix, xp.T).T
        sqnormp = np.sum(xp * Ixp, axis=1)

        sqdistances = sqnormp.reshape(-1, 1) + sqnormp.reshape(1, -1) - 2 * np.matmul(xp, Ixp.T)

        sqdistances = check_sqdist_numerics_diagzero(sqdistances)

        distances = np.sqrt(sqdistances)

    elif space=="sphere":
        distances = dist_sphere_riem_amb(xp, xp, g_matrix)

    else:
        raise ValueError("space option not recognized")

    silh = sklearn.metrics.silhouette_score(distances, labels, metric="precomputed")

    return silh, np.max(distances)


DISTABSTOL = 1e-5
DISTRELTOL = 1e-10
DISTABSTOLDIAG = 1e-2


def check_sqdist_numerics_diagzero(sqdistances):
    diag_dist = np.diag(sqdistances)
    assert np.allclose(diag_dist, 0., atol=DISTABSTOLDIAG), "sqdistances are wrong, found: min {:} ,  max {:}".format(
        np.min(diag_dist), np.max(diag_dist))

    np.fill_diagonal(sqdistances, 0.)
    min_dist = np.min(sqdistances)
    max_dist = np.max(sqdistances)

    all_ok = min_dist >= 0 or -min_dist < DISTABSTOL  # ((max_dist-min_dist)/max_dist - 1 < DISTRELTOL and -min_dist < DISTABSTOL)
    assert all_ok, "sqdistances are wrong, found: min {:} ,  max {:}".format(min_dist, max_dist)

    if min_dist < 0.:
        sqdistances += -np.min(sqdistances)
        np.fill_diagonal(sqdistances, 0.)

    return sqdistances

def rough_sqdist_numerics_diagzero(sqdistances):

    np.fill_diagonal(sqdistances, 0.)
    min_dist = np.min(sqdistances)
    max_dist = np.max(sqdistances)

    if min_dist < 0.:
        sqdistances += -np.min(sqdistances)
        np.fill_diagonal(sqdistances, 0.)

    return sqdistances

def check_sqdist_numerics(sqdistances):
    min_dist = np.min(sqdistances)
    max_dist = np.max(sqdistances)

    all_ok = min_dist >= 0 or -min_dist < DISTABSTOL  # ((max_dist-min_dist)/max_dist - 1 < DISTRELTOL and -min_dist < DISTABSTOL)
    assert all_ok, "sqdistances are wrong, found: min {:} ,  max {:}".format(min_dist, max_dist)

    if min_dist < 0.:
        sqdistances += -np.min(sqdistances)

    return sqdistances

def print_scores(labels, predictions, clusters_points, dist_method):
    scores_list = compute_scores(labels, predictions, clusters_points, dist_method)
    print(scores_list)

def compute_scores(labels, predictions, clusters_points, dist_method):
    distances = dist_method(clusters_points, clusters_points)
    scores_list = [
        purity_score(labels, predictions),
        sklearn.metrics.homogeneity_score(labels, predictions),
        sklearn.metrics.completeness_score(labels, predictions),
        sklearn.metrics.silhouette_score(distances, predictions, metric="precomputed")]
    return scores_list

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = sklearn.metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def sim_score(clusters, centers, g_matrix):
    inertia = [np.mean(dist_on_sphere(cl, cen, g_matrix)) for cl, cen in zip(clusters, centers)]
    return -np.mean(inertia)

def aggregate_dissimilarity(clusters, centers, g_matrix):
    np.testing.assert_equal(len(centers), len(clusters))
    aggr_diss = np.array([len(cl)**2 * (1 - riemannian_sqnorm(ct, g_matrix)) for cl, ct in zip(clusters, centers)]).reshape(-1)
    np.testing.assert_equal(len(aggr_diss), len(clusters))
    return aggr_diss

def cos_similarity(emb1, emb2, g_matrix):
    emb1 = check_and_reshape(emb1)
    emb2 = check_and_reshape(emb2)

    Iemb1 = np.matmul(g_matrix, emb1.T)
    Iemb2 = np.matmul(g_matrix, emb2.T)

    scalprods = np.matmul(emb1, Iemb2)

    norms1 = np.sqrt(np.sum(emb1 * Iemb1.T, axis=1))
    norms2 = np.sqrt(np.sum(emb2 * Iemb2.T, axis=1))

    scalprods = scalprods / (norms1.reshape(-1,1) * norms2.reshape(1,-1))

    return scalprods

def dist_on_sphere(emb1, emb2, g_matrix):
    sims = cos_similarity(emb1, emb2, g_matrix)
    return np.arccos(np.clip(sims, a_min=-1, a_max=1))

def check_and_reshape(emb):
    emb = np.asarray(emb)
    assert (len(emb.shape)==np.array([1,2])).any(), "`emb` must be either a 1D or a 2D ndarray"
    if len(emb.shape)==1:
        emb = emb.reshape(1,-1)
    return emb

def riemannian_norm(points, g_matrix):
    sqnormp = riemannian_sqnorm(points, g_matrix)
    return np.sqrt(sqnormp)

def riemannian_sqnorm(points, g_matrix):
    points = check_and_reshape(points)
    ICp = np.matmul(g_matrix, points.T).T
    sqnormp = np.sum(points * ICp, axis=1).reshape(-1)
    return sqnormp

def riemannian_dist(xp, xq, g_matrix):

    Ixp = np.matmul(g_matrix, xp.T)
    sqnormp = np.sum(xp * Ixp.T, axis=1)

    Ixq = np.matmul(g_matrix, xq.T)
    sqnormq = np.sum(xq * Ixq.T, axis=1)

    sqdistances = sqnormp.reshape(-1, 1) + sqnormq.reshape(1, -1) - 2 * np.matmul(xp, Ixq)

    # # temporarily remove this check put a rough compliance function
    # sqdistances = check_sqdist_numerics_diagzero(sqdistances)
    sqdistances = rough_sqdist_numerics_diagzero(sqdistances)

    distances = np.sqrt(sqdistances)

    return distances

NORMTOL = 1e-15

def riemannian_normalize(points, g_matrix):
    return points / (riemannian_norm(points, g_matrix).reshape(-1, 1) + NORMTOL)
