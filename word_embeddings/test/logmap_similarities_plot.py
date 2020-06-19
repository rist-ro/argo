import os
n_cores = "40"
os.environ["OMP_NUM_THREADS"] = n_cores
os.environ["OPENBLAS_NUM_THREADS"] = n_cores
os.environ["MKL_NUM_THREADS"] = n_cores
os.environ["VECLIB_MAXIMUM_THREADS"] = n_cores
os.environ["NUMEXPR_NUM_THREADS"] = n_cores

import json
import scipy
import scipy.sparse
import scipy.sparse.linalg
import numpy as np
num_plots = 20
from cycler import cycler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core.plotting import *
from .core.load_embeddings import *
from .core.compute_embeddings import *
from .core.spaces import *


# matplotlib.rc('text', usetex=True)
# matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Lucida']})
matplotlib.rcParams.update({'savefig.dpi': '300'})

fontsize=20
fonttitlesize=fontsize
fontaxeslabelsize=16
fontlegendsize=16
matplotlib.rcParams.update({'font.size': fontsize})
# matplotlib.rcParams.update({'font.weight': 'bold'})
# matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

matplotlib.rcParams.update({'legend.frameon': False})
matplotlib.rcParams.update({'legend.fontsize': fontlegendsize})
matplotlib.rcParams.update({'legend.borderaxespad': 1.0})

matplotlib.rcParams.update({'lines.linewidth': 4.0})
matplotlib.rcParams.update({'lines.markersize': 9})

matplotlib.rcParams.update({'axes.titlesize': fonttitlesize})
matplotlib.rcParams.update({'axes.labelsize': fontaxeslabelsize})

colormap = plt.cm.gist_ncar
colors = [colormap(i) for i in np.linspace(0, 0.9, num_plots)]
matplotlib.rcParams.update({'axes.prop_cycle' : cycler(color=colors)})
# plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
# plt.gca().set_prop_cycle('color', colors)

# matplotlib.rcParams.update({'axes.labelpad': 16.0})
# matplotlib.rcParams.update({'xtick.major.pad': 10.0})
# matplotlib.rcParams.update({'ytick.major.pad': 5.0})

figsize = matplotlib.rcParams['figure.figsize']
figsize[0] = 6.4 * 3
figsize[1] = 4.8 * 3

import glob, csv
from scipy.stats import spearmanr
import sys
import readers
import pandas as pd
from spaces import HyperSphere, EuclideanSpace
# import pandas as pd
from mylogging import init_stream_logger, init_logger
stream_logger = init_stream_logger()

import numexpr as ne
import gc
import argparse
# import memory_profiler


def evaluate_similarities(filter_dictionary, simeval, logger, datasets=[], outbasename=None, plot=True):
    """Evaluate the trained word vectors on a variety of similarity files

    Args:
        simeval (method): a method that takes two lists of words and returns a similarity measure. simeval(words1,words2),
                        i.e. simeval(["dog", "cat"], ["tiger", "airplane"])
        # embeddings_manager (spaces.EmbeddingsManager): embeddings manager, to analyze embeddings.
        logger (Logger): class to use for logging
    """

    folder = '/data/captamerica_hd2/text/similarities_datasets/splitted'

    # # in each of these folders I expect to find the splits
    # datasets = [
    #         "wordsim353", "mc", "rg", "scws",
    #         "wordsim353sim", "wordsim353rel",
    #         "men", "mturk287", "rw", "simlex999"
    #         ]

    filenames = []

    for dirname in datasets:
        fsplits = glob.glob(os.path.join(folder, os.path.join(dirname, "*.csv")))
        filenames.append(sorted(fsplits))

    filenames = list(itertools.chain.from_iterable(filenames))

    scores_dict = {}
    corr_dict = {}

    all_humscores = []
    all_simscores = []
    plot_method = "scatter"
    title = ""
    xlabel = "human scores"
    ylabel = "similarity"

    full_count = 0
    count_tot = 0

    for i in range(len(filenames)):

        label = "-".join(os.path.splitext(filenames[i])[0].split("/")[-2:])

        with open(filenames[i], 'r') as instream:
            reader = csv.reader(instream, delimiter=' ', skipinitialspace=True)
            lines = list(reader)
            full_count += len(lines)
            words1, words2, humscores = zip(
                *((w1, w2, sc) for (w1, w2, sc) in lines if (w1 in filter_dictionary) and (w2 in filter_dictionary)))
            count_tot += len(words1)

        #     full_data = [line.rstrip().split(' ') for line in f]
        #     full_count += len(full_data)
        #     data = [x for x in full_data if all(word in dictionary for word in x)]
        #
        # indices = np.array([[dictionary[word] for word in row] for row in data])
        # ind1, ind2, ind3, ind4 = indices.T
        # read csv in a table and then calculate the distances and pair them with the scores from the csv
        humscores = np.array(humscores, dtype=np.float)
        simscores = simeval(words1, words2)

        if plot:
            plot_dict = create_plot_data_dict([humscores], [simscores], [label],
                                              xlabel, ylabel, plot_method_name=plot_method)

            axes = initialize_plot(title, xlabel, ylabel)
            plot_data(plot_dict["data"], axes, plot_method, plot_args=[], plot_kwargs={})

            if outbasename:
                outputfile = outbasename + "-" + label + ".dat"
                save_object(plot_dict, outputfile)
                outputpng = outbasename + "-" + label + ".png"
                finalize_plot(axes, outputpng, xlim=None, ylim=None, legendloc=None)
            else:
                plt.show()

        all_humscores += list(humscores)
        all_simscores += list(simscores)
        scores_dict[label] = (humscores, simscores)

        corr, p_value = spearmanr(humscores, simscores)
        corr = corr * 100
        corr_dict[label] = corr

        logger.info("%s:" % filenames[i])
        logger.info('SPEARMAN CORR: %.2f ' % corr)

    logger.info('Questions seen/total: %.2f%% (%d/%d)' %
                (100 * count_tot / float(full_count), count_tot, full_count))

    logger.info('\nON ALL')
    logger.info('-------------------------')
    corr, p_value = spearmanr(all_humscores, all_simscores)
    corr = corr * 100

    label = "all"
    corr_dict[label] = corr

    logger.info('SPEARMAN CORR: %.2f ' % corr)
    logger.info('\n')

    if plot:
        plot_dict = create_plot_data_dict([all_humscores], [all_simscores], ["all"],
                                          xlabel, ylabel, plot_method_name=plot_method)

        axes = initialize_plot(title, xlabel, ylabel)
        plot_data(plot_dict["data"], axes, plot_method, plot_args=[], plot_kwargs={})

        if outbasename:
            outputfile = outbasename + "-all.dat"
            save_object(plot_dict, outputfile)
            outputpng = outbasename + "-all.png"
            finalize_plot(axes, outputpng, xlim=None, ylim=None, legendloc=None)
        else:
            plt.show()

    return scores_dict, corr_dict



def read_cooccurrences_from_c(filename):
    dt = np.dtype("i4, i4, f8")
    cooccurrences = np.array(np.fromfile(filename, dtype=dt))

    row_ind = cooccurrences[:]['f0']-1
    col_ind = cooccurrences[:]['f1']-1
    values = cooccurrences[:]['f2']

    D = max(max(row_ind[-5:]), max(col_ind[-5:])) + 1
    print(D)
    C = scipy.sparse.coo_matrix((values, (row_ind, col_ind)), shape=(D, D))
    C = scipy.sparse.csr_matrix(C)
    return C


def read_cooccurrences_from_c_old(filename):
    # import cooccurrences with couple counts
    from ctypes import Structure, sizeof, c_int, c_double

    class CREC(Structure):
        _fields_ = [('w1', c_int),
                    ('w2', c_int),
                    ('value', c_double)]

    with open(filename, 'rb') as stream:
        cooccurrences = []
        cr = CREC()
        while stream.readinto(cr) == sizeof(cr):
            cooccurrences.append((cr.w1-1, cr.w2-1, cr.value))

    row_ind, col_ind, values = zip(*cooccurrences)

    D = max(row_ind+col_ind)+1
    print(D)
    C = scipy.sparse.coo_matrix((values, (row_ind, col_ind)), shape = (D, D))
    
    return C


def sizeof_fmt(num, suffix='B'):
    ''' By Fred Cirera, after https://stackoverflow.com/a/1094933/1870254'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def see_all_vars(howmany=None):
    for name, size in sorted(((name, sys.getsizeof(value)) for name,value in locals().items()),
                             key= lambda x: -x[1])[:howmany]:
        print("{:>30} : {:>8}".format(name,sizeof_fmt(size)))



def similarity_KL(embeddings, dictionary, words1, words2, symm=True):
    emb1 = get_word_embeddings(words1, embeddings, dictionary)
    emb2 = get_word_embeddings(words2, embeddings, dictionary)

    sims = KL(emb1, emb2)
    if symm:
        sims += KL(emb2, emb1)
        sims /= 2.

    return -sims


def similarity_BC(embeddings, dictionary, words1, words2):
    emb1 = get_word_embeddings(words1, embeddings, dictionary)
    emb2 = get_word_embeddings(words2, embeddings, dictionary)

    sims = BC(emb1, emb2)

    return -sims

def divergence(p, q, alpha):
    p += NUMTOL
    q += NUMTOL
    div = 0
    if alpha == 1:
        div = np.sum(p - q + q * np.log(q / p), axis=-1)
    elif alpha == -1:
        div = np.sum(q - p + p * np.log(p / q), axis=-1)
    else:
        c0 = 4 / (1 - alpha ** 2)
        cp = (1 - alpha) / 2
        cq = (1 + alpha) / 2
        div = np.sum(c0 * (cp * p + cq * q - (p ** cp) * (q ** cq)), axis=-1)

    return div


def similarity_div(embeddings, dictionary, words1, words2, alpha, symm=True):
    emb1 = get_word_embeddings(words1, embeddings, dictionary)
    emb2 = get_word_embeddings(words2, embeddings, dictionary)

    sims = divergence(emb1, emb2, alpha)
    if symm:
        sims += divergence(emb2, emb1, alpha)
        sims /= 2.

    return -sims


def similarity_logmap(embeddings, dictionary, words1, words2, alpha, p0=None, dual_transp=False, metric="alpha", project=False):
    emb1 = get_word_embeddings(words1, embeddings, dictionary)
    emb2 = get_word_embeddings(words2, embeddings, dictionary)

    emb1+=NUMTOL
    emb2+=NUMTOL

    if p0 is None:
        # UNIFORM
        s = embeddings.shape[1]
        p0 = np.ones(s)/s
        # UNIFORM

    p0 = p0.reshape(1, -1)

    ha_p0 = h(p0, alpha)
    ha_emb1 = h(emb1, alpha)
    logmaps01 = ha_emb1 - ha_p0

    if dual_transp:
        ha_emb2 = h(emb2, -alpha)
        ha_x0 = h(p0, -alpha)
        logmaps02 = ha_emb2 - ha_x0
    else:
        ha_emb2 = h(emb2, alpha)
        logmaps02 = ha_emb2 - ha_p0

    # if I am using the metric in the tangent space alpha
    if metric=="alpha":
        g_vec = p0**alpha
        # g = np.diag(g.reshape(-1))
    elif metric=="simplex":
        g_vec = 1/p0
        # g = np.diag(g.reshape(-1))
    elif metric=="id":
        g_vec = 1
        # g = np.eye(p0.shape[1])
    else:
        raise ValueError("metric type not recognized `%s`"%metric)

    if project:
        logmaps01 = project_vectors_away_from_normal(logmaps01, p0, alpha)
        logmaps02 = project_vectors_away_from_normal(logmaps02, p0, alpha)

    # scalprods = riemannian_cosprod(logmaps01, logmaps02, g, normalize=True)
    scalprods = riemannian_cosprod_fordiag(logmaps01, logmaps02, g_vec, normalize=True)
    # scalprods = np.sum(logmaps01 * g * logmaps02, axis=1)

    return scalprods


def similarity_logmap_Esubmodel(p_embeddings, dictionary, words1, words2, alpha, I_inv, beta,
                                p, I_prod, I_norm=None, rescale=False, method="cosprod"):
    # proj logmap
    p1 = get_word_embeddings(words1, p_embeddings, dictionary)
    # u1 = get_word_embeddings(words1, u_embeddings, dictionary)
    # v1 = get_word_embeddings(words1, v_embeddings, dictionary)

    p2 = get_word_embeddings(words2, p_embeddings, dictionary)
    # u2 = get_word_embeddings(words2, u_embeddings, dictionary)
    # v2 = get_word_embeddings(words2, v_embeddings, dictionary)

    p1+=NUMTOL
    p2+=NUMTOL

    p = p.reshape(1, -1)

    ha_p0 = h(p, alpha)
    ha_emb1 = h(p1, alpha)
    logmaps01 = ha_emb1 - ha_p0
    #now for each line need to project
    C_proj1 = project_vectors_on_basis(logmaps01, beta, I_inv, p, alpha)

    ha_emb2 = h(p2, alpha)
    logmaps02 = ha_emb2 - ha_p0
    #now for each line need to project
    C_proj2 = project_vectors_on_basis(logmaps02, beta, I_inv, p, alpha)


    # normalize the vector with which metric, TODO (multiply by a scalar)
    if I_norm is not None:
        C_proj1 = riemannian_normalize(C_proj1, I_norm)
        C_proj2 = riemannian_normalize(C_proj2, I_norm)

    if rescale:
        x1 = h(p1, 0)
        x2 = h(p2, 0)
        x0 = h(p, 0)
        # np.allclose(np.sqrt(np.sum(x1 ** 2, axis=1)), 2)
        # np.allclose(np.sqrt(np.sum(x2 ** 2, axis=1)), 2)
        # np.allclose(np.sqrt(np.sum(x0 ** 2, axis=1)), 2)
        mod1 = 2*np.arccos(np.sum(x1 * x0, axis=1)/4.)
        mod2 = 2*np.arccos(np.sum(x2 * x0, axis=1)/4.)
        C_proj1 = C_proj1 * mod1.reshape(-1, 1)
        C_proj2 = C_proj2 * mod2.reshape(-1, 1)

    if method == "cos":
        #scalar product with which metric
        scalprods = riemannian_cosprod(C_proj1, C_proj2, I_prod, normalize=False)
    elif method == "dis":
        scalprods = -riemannian_distance(C_proj1, C_proj2, I_prod)
    else:
        raise ValueError("expected only `cos` or `dis`, instead %s given."%method)
    return scalprods

def similarity_logmap_Esubmodel_trick(p_embeddings, dictionary, words1, words2, alpha, I_inv, DV,
                                p, I_prod, I_norm=None, rescale=False, method="cos"):
    # proj logmap
    p1 = get_word_embeddings(words1, p_embeddings, dictionary)
    # u1 = get_word_embeddings(words1, u_embeddings, dictionary)
    # v1 = get_word_embeddings(words1, v_embeddings, dictionary)

    p2 = get_word_embeddings(words2, p_embeddings, dictionary)
    # u2 = get_word_embeddings(words2, u_embeddings, dictionary)
    # v2 = get_word_embeddings(words2, v_embeddings, dictionary)

    p1+=NUMTOL
    p2+=NUMTOL

    p = p.reshape(1, -1)

    # ha_p0 = h(p, alpha)
    # ha_emb1 = h(p1, alpha)
    # logmaps01 = ha_emb1 - ha_p0
    # #now for each line need to project
    C_proj1 = project_on_basis_from_ps(p1, DV, I_inv, p, alpha)

    # ha_emb2 = h(p2, alpha)
    # logmaps02 = ha_emb2 - ha_p0
    #now for each line need to project
    C_proj2 = project_on_basis_from_ps(p2, DV, I_inv, p, alpha)


    # normalize the vector with which metric, TODO (multiply by a scalar)
    if I_norm is not None:
        C_proj1 = riemannian_normalize(C_proj1, I_norm)
        C_proj2 = riemannian_normalize(C_proj2, I_norm)

    if rescale:
        x1 = h(p1, 0)
        x2 = h(p2, 0)
        x0 = h(p, 0)
        # np.allclose(np.sqrt(np.sum(x1 ** 2, axis=1)), 2)
        # np.allclose(np.sqrt(np.sum(x2 ** 2, axis=1)), 2)
        # np.allclose(np.sqrt(np.sum(x0 ** 2, axis=1)), 2)
        mod1 = 2*np.arccos(np.sum(x1 * x0, axis=1)/4.)
        mod2 = 2*np.arccos(np.sum(x2 * x0, axis=1)/4.)
        C_proj1 = C_proj1 * mod1.reshape(-1, 1)
        C_proj2 = C_proj2 * mod2.reshape(-1, 1)

    if method == "cos":
        #scalar product with which metric
        scalprods = riemannian_cosprod(C_proj1, C_proj2, I_prod, normalize=False)
    elif method == "dis":
        scalprods = -riemannian_distance(C_proj1, C_proj2, I_prod)
    else:
        raise ValueError("expected only `cos` or `dis`, instead %s given."%method)
    return scalprods


def similarity_logmap_hyps(embeddings, dictionary, words1, words2, x0=None):
    hyps = HyperSphere(embeddings.shape[1] - 1)
    emb1 = get_word_embeddings(words1, embeddings, dictionary)
    emb2 = get_word_embeddings(words2, embeddings, dictionary)

    if x0 is None:
        # UNIFORM
        x0 = hyps._x0
        # UNIFORM

        # # MEAN
        # x0 = hyps.mean(embeddings_manager.embeddings)
        # # MEAN

        # # MEDIAN
        # x0 = hyps.median(embeddings_manager.embeddings)
        # # MEDIAN

    logmaps01 = np.array([hyps.logmap(x0, xa) for xa in emb1])
    logmaps02 = np.array([hyps.logmap(x0, xb) for xb in emb2])

    # #PROJECT LOGMAPS ON THE SUBMODEL TANGENT SPACE
    # u_embeddings, v_embeddings = embeddings_manager.extra_info
    # v_embeddings_normalized = v_embeddings / np.linalg.norm(v_embeddings, axis=0)
    #
    # prods_log10_V = np.matmul(logmaps10, v_embeddings_normalized)
    # projected_logmaps10 = np.matmul(prods_log10_V, np.transpose(v_embeddings_normalized))
    #
    # prods_log20_V = np.matmul(logmaps20, v_embeddings_normalized)
    # projected_logmaps20 = np.matmul(prods_log20_V, np.transpose(v_embeddings_normalized))

    # np.sum(np.transpose(np.transpose(projected_logmaps10)*np.transpose(logmaps10)/(np.linalg.norm(projected_logmaps10, axis=1)*np.linalg.norm(logmaps10, axis=1))),axis=1)

    # #LINEARDIFF
    # logmaps10 = emb1-x0
    # logmaps20 = emb2-x0
    # #LINEARDIFF

    # logmaps10=projected_logmaps10
    # logmaps20=projected_logmaps20
    scalprods = np.sum(logmaps01 * logmaps02, axis=1) / (
                np.linalg.norm(logmaps01, axis=1) * np.linalg.norm(logmaps02, axis=1))
    # scalprods = np.linalg.norm(logmaps20-logmaps10, axis=1)

    return scalprods

def similarity_cosprod(embeddings, dictionary, words1, words2, normalize=True):
    emb1 = get_word_embeddings(words1, embeddings, dictionary)
    emb2 = get_word_embeddings(words2, embeddings, dictionary)
    scalprods = np.sum(emb1*emb2, axis=1)
    if normalize:
        scalprods = scalprods / (np.linalg.norm(emb1, axis=1) * np.linalg.norm(emb2, axis=1))
    return scalprods

def similarity_cos_fisher(embeddings, dictionary, words1, words2, I, normalize=False):
    emb1 = get_word_embeddings(words1, embeddings, dictionary)
    emb2 = get_word_embeddings(words2, embeddings, dictionary)

    return riemannian_cosprod(emb1, emb2, I, normalize=normalize)

    # Iemb1 = np.matmul(I, emb1.T).T
    # scalprods = np.sum(emb2*Iemb1, axis=1)
    #
    # if normalize:
    #     Iemb2 = np.matmul(I, emb2.T).T
    #     norms1 = np.sqrt(np.sum(emb1 * Iemb1, axis=1))
    #     norms2 = np.sqrt(np.sum(emb2 * Iemb2, axis=1))
    #     scalprods = scalprods / (norms1*norms2)
    #
    # return scalprods

def riemannian_distance(Ca, Cb, I):
    #a is a list of vectors
    #b is a list of vectors
    #I is the metric

    vec = Cb-Ca

    Ivec = np.matmul(I, vec.T).T
    distances = np.sum(vec * Ivec, axis=1)

    return distances

def riemannian_cosprod(Ca, Cb, I, normalize=True):
    #a is a list of vectors
    #b is a list of vectors
    #I is the metric
    np.testing.assert_array_equal([len(Ca.shape), len(Cb.shape)], [2,2])
    needed_I_shape = [Ca.shape[1], Cb.shape[1]]
    np.testing.assert_array_equal(needed_I_shape, I.shape)

    ICb = np.matmul(I, Cb.T).T
    scalprods = np.sum(Ca * ICb, axis=1)

    if normalize:
        ICa = np.matmul(I, Ca.T).T
        norms1 = np.sqrt(np.sum(Ca * ICa, axis=1))
        norms2 = np.sqrt(np.sum(Cb * ICb, axis=1))
        scalprods = scalprods / (norms1*norms2)

    return scalprods

def riemannian_cosprod_fordiag(Ca, Cb, g_vec, normalize=True):
    #a is a list of vectors
    #b is a list of vectors
    #I is the metric

    if isinstance(g_vec, np.ndarray):
        g_vec = g_vec.reshape(1,-1)

    ICb = g_vec*Cb
    scalprods = np.sum(Ca * ICb, axis=1)

    if normalize:
        ICa = g_vec*Ca
        norms1 = np.sqrt(np.sum(Ca * ICa, axis=1))
        norms2 = np.sqrt(np.sum(Cb * ICb, axis=1))
        scalprods = scalprods / (norms1*norms2)

    return scalprods

def riemannian_normalize(C, I):
    IC = np.matmul(I, C.T).T
    norms = np.sqrt(np.sum(C * IC, axis=1))
    Cnorm = C / norms.reshape(-1, 1)
    return Cnorm

def similarity_fisher_uv(u_embeddings, v_embeddings, embs_name, dictionary, correlations={}, y_data={},
                         filter_dictionary=None):
    # SIMILARITIES FISHER U V
    scores = {}
    if filter_dictionary is None:
        filter_dictionary = dictionary

    p0 = np.ones(v_embeddings.shape[0])/v_embeddings.shape[0]
    p0 = p0.reshape(-1,1)
    I, _ = fisher_matrix_and_whatnot(v_embeddings, p0)

    method_name = embs_name + "-0-f"
    def simeval(words1, words2):
        return similarity_cos_fisher(u_embeddings, dictionary, words1, words2, I)

    similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data)

    method_name = embs_name + "-0-i_n"
    def simeval(words1, words2):
        return similarity_cos_fisher(u_embeddings, dictionary, words1, words2, np.eye(v_embeddings.shape[1]), normalize=True)

    similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data)

    method_name = embs_name + "-0-f_n"
    def simeval(words1, words2):
        return similarity_cos_fisher(u_embeddings, dictionary, words1, words2, I, normalize=True)

    similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data)

    return correlations, y_data


def similarity_cosUVmod(U, V, dictionary, words1, words2, p0=None, normalize=True):
    emb1 = get_word_embeddings(words1, U, dictionary)
    emb2 = get_word_embeddings(words2, U, dictionary)

    V = V[:-1, :]

    if p0 is None:
        # UNIFORM
        D = V.shape[0]
        p0 = np.ones(D)/D
        # UNIFORM

    p0 = p0.reshape([-1, 1])
    g = np.matmul(V.T, ne.evaluate("p0 * V"))

    emb2 = np.matmul(emb2, g)
    scalprods = ne.evaluate("sum(emb1 * emb2, axis=1)")
    if normalize:
        scalprods = scalprods / (np.linalg.norm(emb1, axis=1) * np.linalg.norm(emb2, axis=1))
    return scalprods

def similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, extra_param_name=""):
    full_name = method_name + extra_param_name

    stream_logger.info(full_name)
    stream_logger.info("---------------------------------------------------------------------------")
    sc, corr = evaluate_similarities(filter_dictionary, simeval, stream_logger, datasets = all_datasets_names, plot=False)
    stream_logger.info("---------------------------------------------------------------------------")
    stream_logger.info("")

    scores[full_name] = sc
    correlations[full_name] = corr

    for task_name in corr:
        if y_data.get(task_name, None) is None:
            y_data[task_name] = {}
        if y_data[task_name].get(method_name, None) is None:
            y_data[task_name][method_name] = []

        y_data[task_name][method_name].append(corr[task_name])

ymin = 0
ymax = 100

import re
def get_number_from_string(string):
    floats_str = re.findall('\-?\d+\.\d+', string)

    if len(floats_str)==0:
        return -np.inf
    elif len(floats_str)==1:
        return float(floats_str[0])
    else:
        raise ValueError("found more than one float `%s` in the string `%s`"%(str(floats_str), string))


all_datasets_names = [
            "wordsim353",
            "mc", "rg", "scws",
            "wordsim353sim", "wordsim353rel",
            "men", "mturk287", "rw", "simlex999"
            ]

ordered_columns = []
n_splits = 3

for dsname in all_datasets_names:
    for i in range(n_splits):
        ordered_columns.append(dsname+"-split_{:d}".format(i))


def print_table(correlations, outputfile = None):
    logger = stream_logger if outputfile is None else init_logger(outputfile)

    df = pd.DataFrame(correlations).transpose()[ordered_columns]
    df['indexNumber'] = [get_number_from_string(i) for i in df.index]
    df['indexes'] = df.index

    df.sort_values(['indexNumber', 'indexes'], ascending=True, inplace=True)
    df.drop(['indexNumber', 'indexes'], axis=1, inplace=True)

    logger.info("---------------------------------------------------------------------------")
    logger.info(df.to_string())
    logger.info("---------------------------------------------------------------------------")
    logger.info("")

def get_style_code(method_name):
    if "-f" in method_name or "-a" in method_name:
        stylestr = '.'
    elif "-i" in method_name:
        stylestr = 'x'
    else:
        return ':'

    if "-0-" in method_name:
        stylestr+= '--'
    elif "-u-" in method_name:
        stylestr += '-'

    return stylestr

hline_color = {
              "enwiki-u+v-n-cosprod" : "0.1",
              "swiki-u+v-n-cosprod" : "0.1",
              "wikigiga5-u+v-n-cosprod" : "0.3",
              "p_cIw-cn-cosprod" : "0.5",
              "enwiki-u-cosprod" : "0.7",
              "enwiki-u-n-cosprod" : "0.9",
              "swiki-u-cosprod" : "0.7",
              "swiki-u-n-cosprod" : "0.9"}

def plot_all_tasks(alphas, y_data, outname):
    xmin = min(alphas)
    xmax = max(alphas)

    # y_data is a dictionary of dictionaries of list
    # task -> method -> list of accuracies corresponding to the parameters (alphas)
    for task_name in y_data:
        plot_name = os.path.basename(outname)
        axes = initialize_plot(plot_name, "alpha", "")

        y_all_methods = y_data[task_name]
        for method_name in sorted(y_all_methods.keys()):
            y = y_all_methods[method_name]

            if len(y)==0:
                raise ValueError("y_data of %s is empty..."%method_name)
            elif len(y)==1:
                plt.hlines(y[0], xmin, xmax, linestyle='--', linewidth=2, color=hline_color[method_name], label=method_name)
            else:
                stylecodestr = get_style_code(method_name)
                plt.plot(alphas, y, stylecodestr, label=method_name)

        finalize_plot(axes, outname+"-"+task_name+".png", xlim=(xmin,xmax), ylim=(ymin,ymax))

    with open(outname+".json", 'w') as fstream:
        json.dump(y_data, fstream)

    plt.close('all')


def similarities_divergences(p_cIw, prob_name, alphas, dictionary, y_data={}, correlations={}, filter_dictionary=None):
    # SIMILARITIES DIVERGENCES
    scores = {}

    if filter_dictionary is None:
        filter_dictionary = dictionary

    # KL between p(c|w)
    method_name = prob_name+"-KL"
    def simeval(words1, words2):
        return similarity_KL(p_cIw, dictionary, words1, words2, symm=False)

    similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data)

    # KL between p(c|w) symmetric
    method_name = prob_name+"-KLsymm"
    def simeval(words1, words2):
        return similarity_KL(p_cIw, dictionary, words1, words2, symm=True)

    similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data)

    # Bhattharchayya between p(c|w)
    method_name = prob_name + "-BC"

    def simeval(words1, words2):
        return similarity_BC(p_cIw, dictionary, words1, words2)

    similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data)

    for a in alphas:
        # alpha divergence between p(c|w)
        method_name = prob_name+"-D"
        def simeval(words1, words2):
            return similarity_div(p_cIw, dictionary, words1, words2, a, symm=False)

        similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))


        method_name = prob_name+"-Dsymm"
        def simeval(words1, words2):
            return similarity_div(p_cIw, dictionary, words1, words2, a, symm=True)

        similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))


    return correlations, y_data


def similarities_logmaps_fullrank(p_cIw, prob_name, p_w, alphas, dictionary, correlations={}, y_data={}, filter_dictionary=None):

    # SIMILARITIES LOGMAPS
    scores = {}
    if filter_dictionary is None:
        filter_dictionary = dictionary

    for a in alphas:

        method_name = prob_name + "-0-Log-a"
        def simeval(words1, words2):
            return similarity_logmap(p_cIw, dictionary, words1, words2, a, metric="alpha")
        similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))

        # in the uniform p0, all metrics: alpha, simplex and id are equivalent in terms of similarity score since they are one multiple of the other
        # method_name = prob_name + "-0-Log-i"
        # def simeval(words1, words2):
        #     return similarity_logmap(p_cIw, dictionary, words1, words2, a, metric="id")
        # similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))
        #
        # method_name = prob_name + "-0-Log-s"
        # def simeval(words1, words2):
        #     return similarity_logmap(p_cIw, dictionary, words1, words2, a, metric="simplex")
        # similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))


        # method_name = prob_name+"-0-DTLog"
        # def simeval(words1, words2):
        #     return similarity_logmap(p_cIw, dictionary, words1, words2, a, dual_transp=True)
        #
        # similarity_calc_and_org(method_name, simeval, scores, correlations, y_data, str(a))

        method_name = prob_name+"-u-Log-a"
        def simeval(words1, words2):
            return similarity_logmap(p_cIw, dictionary, words1, words2, a, p0=p_w, metric="alpha")
        similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))

        method_name = prob_name+"-u-Log-s"
        def simeval(words1, words2):
            return similarity_logmap(p_cIw, dictionary, words1, words2, a, p0=p_w, metric="simplex")
        similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))

        method_name = prob_name+"-u-Log-i"
        def simeval(words1, words2):
            return similarity_logmap(p_cIw, dictionary, words1, words2, a, p0=p_w, metric="id")
        similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))


        method_name = prob_name + "-0-pjLog-a"
        def simeval(words1, words2):
            return similarity_logmap(p_cIw, dictionary, words1, words2, a, metric="alpha", project=True)
        similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))

        method_name = prob_name+"-u-pjLog-a"
        def simeval(words1, words2):
            return similarity_logmap(p_cIw, dictionary, words1, words2, a, p0=p_w, metric="alpha", project=True)
        similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))

        method_name = prob_name+"-u-pjLog-s"
        def simeval(words1, words2):
            return similarity_logmap(p_cIw, dictionary, words1, words2, a, p0=p_w, metric="simplex", project=True)
        similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))

        method_name = prob_name+"-u-pjLog-i"
        def simeval(words1, words2):
            return similarity_logmap(p_cIw, dictionary, words1, words2, a, p0=p_w, metric="id", project=True)
        similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))

        # method_name = prob_name+"-u-DTLog"
        # def simeval(words1, words2):
        #     return similarity_logmap(p_cIw, dictionary, words1, words2, a, x0=p_w, dual_transp=True)
        #
        # similarity_calc_and_org(method_name, simeval, scores, correlations, y_data, str(a))

    return correlations, y_data

def similarities_logmaps_Esubmodel(p_cIw, U, V, prob_name, pu, alphas, dictionary,
                                   correlations={}, y_data={}, filter_dictionary=None, method="cos"):

    # SIMILARITIES LOGMAPS
    scores = {}
    if filter_dictionary is None:
        filter_dictionary = dictionary

    # I is the fisher matrix, beta is the (pushforward of a vector in u -> to a vector in R^n_(alpha))
    # i.e. matrix containing as rows the coordinates of the basis for the fisher matrix (beta) in R^n_(alpha)

    p0 = np.ones(V.shape[0]) / V.shape[0]
    p0 = p0.reshape(-1, 1)
    I0, DV0 = fisher_matrix_and_whatnot(V, p0)
    I0_inv = np.linalg.inv(I0)

    pu = pu.reshape(-1, 1)
    Iu, DVu = fisher_matrix_and_whatnot(V, pu)
    Iu_inv = np.linalg.inv(Iu)

    Id = np.eye(V.shape[1])

    for a in alphas:
        beta0 = beta_fisher_basis(DV0, p0, a)
        betau = beta_fisher_basis(DVu, pu, a)

        # method_name = prob_name+"-0-pjLog-f"+method
        # def simeval(words1, words2):
        #     return similarity_logmap_Esubmodel(p_cIw, dictionary, words1, words2, a, I0_inv, beta0, p0, I0, method=method)
        # similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))

        method_name = prob_name+"-0-pjLog-nf-f"+method
        def simeval(words1, words2):
            return similarity_logmap_Esubmodel(p_cIw, dictionary, words1, words2, a, I0_inv, beta0, p0, I0, I_norm=I0, method=method)
        similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))

        # method_name = prob_name+"-0-pjLog-nf-r-f"+method
        # def simeval(words1, words2):
        #     return similarity_logmap_Esubmodel(p_cIw, dictionary, words1, words2, a, I0_inv, beta0, p0, I0, I_norm=I0, rescale=True, method=method)
        # similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))
        #
        # method_name = prob_name+"-0-pjLog-nf-r-i"+method
        # def simeval(words1, words2):
        #     return similarity_logmap_Esubmodel(p_cIw, dictionary, words1, words2, a, I0_inv, beta0, p0, Id, I_norm=I0, rescale=True, method=method)
        # similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))

        method_name = prob_name+"-0-pjLog-nf-i"+method
        def simeval(words1, words2):
            return similarity_logmap_Esubmodel(p_cIw, dictionary, words1, words2, a, I0_inv, beta0, p0, Id, I_norm=I0, method=method)
        similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))

        method_name = prob_name+"-0-pjLog-ni-i"+method
        def simeval(words1, words2):
            return similarity_logmap_Esubmodel(p_cIw, dictionary, words1, words2, a, I0_inv, beta0, p0, Id, I_norm=Id, method=method)
        similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))

        # method_name = prob_name+"-0-pjLog-i"+method
        # def simeval(words1, words2):
        #     return similarity_logmap_Esubmodel(p_cIw, dictionary, words1, words2, a, I0_inv, beta0, p0, Id, method=method)
        # similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))
        #
        # method_name = prob_name+"-0-pjLog-ni-f"+method
        # def simeval(words1, words2):
        #     return similarity_logmap_Esubmodel(p_cIw, dictionary, words1, words2, a, I0_inv, beta0, p0, I0, I_norm=Id, method=method)
        # similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))

        method_name = prob_name + "-u-pjLog-nf-f"+method
        def simeval(words1, words2):
            return similarity_logmap_Esubmodel(p_cIw, dictionary, words1, words2, a, Iu_inv, betau, pu, Iu, I_norm=Iu, method=method)
        similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))

        # method_name = prob_name + "-u-pjLog-f"+method
        # def simeval(words1, words2):
        #     return similarity_logmap_Esubmodel(p_cIw, dictionary, words1, words2, a, Iu_inv, betau, pu, Iu, method=method)
        # similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))
        #
        # method_name = prob_name + "-u-pjLog-nf-r-f"+method
        # def simeval(words1, words2):
        #     return similarity_logmap_Esubmodel(p_cIw, dictionary, words1, words2, a, Iu_inv, betau, pu, Iu, I_norm=Iu, rescale=True, method=method)
        # similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))
        #
        # method_name = prob_name + "-u-pjLog-nf-r-i"+method
        # def simeval(words1, words2):
        #     return similarity_logmap_Esubmodel(p_cIw, dictionary, words1, words2, a, Iu_inv, betau, pu, Id, I_norm=Iu, rescale=True, method=method)
        # similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))

        method_name = prob_name + "-u-pjLog-nf-i"+method
        def simeval(words1, words2):
            return similarity_logmap_Esubmodel(p_cIw, dictionary, words1, words2, a, Iu_inv, betau, pu, Id, I_norm=Iu, method=method)
        similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))

        method_name = prob_name + "-u-pjLog-ni-i"+method
        def simeval(words1, words2):
            return similarity_logmap_Esubmodel(p_cIw, dictionary, words1, words2, a, Iu_inv, betau, pu, Id, I_norm=Id, method=method)
        similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))

        # method_name = prob_name + "-u-pjLog-i"+method
        # def simeval(words1, words2):
        #     return similarity_logmap_Esubmodel(p_cIw, dictionary, words1, words2, a, Iu_inv, betau, pu, Id, method=method)
        # similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))
        #
        # method_name = prob_name + "-u-pjLog-ni-f"+method
        # def simeval(words1, words2):
        #     return similarity_logmap_Esubmodel(p_cIw, dictionary, words1, words2, a, Iu_inv, betau, pu, Iu, I_norm=Id, method=method)
        # similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))

    return correlations, y_data

def similarities_logmaps_Esubmodel_trick(p_cIw, U, V, prob_name, pu, alphas, dictionary,
                                   correlations={}, y_data={}, filter_dictionary=None, method="cos"):

    # SIMILARITIES LOGMAPS
    scores = {}
    if filter_dictionary is None:
        filter_dictionary = dictionary

    # I is the fisher matrix, beta is the (pushforward of a vector in u -> to a vector in R^n_(alpha))
    # i.e. matrix containing as rows the coordinates of the basis for the fisher matrix (beta) in R^n_(alpha)

    p0 = np.ones(V.shape[0]) / V.shape[0]
    p0 = p0.reshape(-1, 1)
    I0, DV0 = fisher_matrix_and_whatnot(V, p0)
    I0_inv = np.linalg.inv(I0)

    pu = pu.reshape(-1, 1)
    Iu, DVu = fisher_matrix_and_whatnot(V, pu)
    Iu_inv = np.linalg.inv(Iu)

    Id = np.eye(V.shape[1])

    for a in alphas:

        method_name = prob_name+"-0-pjLog-nf-f"+method
        def simeval(words1, words2):
            return similarity_logmap_Esubmodel_trick(p_cIw, dictionary, words1, words2, a, I0_inv, DV0, p0, I0, I_norm=I0, method=method)
        similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))

        method_name = prob_name+"-0-pjLog-nf-i"+method
        def simeval(words1, words2):
            return similarity_logmap_Esubmodel_trick(p_cIw, dictionary, words1, words2, a, I0_inv, DV0, p0, Id, I_norm=I0, method=method)
        similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))

        method_name = prob_name+"-0-pjLog-ni-i"+method
        def simeval(words1, words2):
            return similarity_logmap_Esubmodel_trick(p_cIw, dictionary, words1, words2, a, I0_inv, DV0, p0, Id, I_norm=Id, method=method)
        similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))

        method_name = prob_name + "-u-pjLog-nf-f"+method
        def simeval(words1, words2):
            return similarity_logmap_Esubmodel_trick(p_cIw, dictionary, words1, words2, a, Iu_inv, DVu, pu, Iu, I_norm=Iu, method=method)
        similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))

        method_name = prob_name + "-u-pjLog-nf-i"+method
        def simeval(words1, words2):
            return similarity_logmap_Esubmodel_trick(p_cIw, dictionary, words1, words2, a, Iu_inv, DVu, pu, Id, I_norm=Iu, method=method)
        similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))

        method_name = prob_name + "-u-pjLog-ni-i"+method
        def simeval(words1, words2):
            return similarity_logmap_Esubmodel_trick(p_cIw, dictionary, words1, words2, a, Iu_inv, DVu, pu, Id, I_norm=Id, method=method)
        similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data, str(a))

    return correlations, y_data

def similarity_euclidean(embs, embs_name, dictionary, correlations={}, y_data={}, filter_dictionary=None):
    # SIMILARITIES EUCLIDEAN
    scores = {}
    if filter_dictionary is None:
        filter_dictionary = dictionary

    method_name = embs_name + "-cosprod"

    def simeval(words1, words2):
        return similarity_cosprod(embs, dictionary, words1, words2)

    similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data)

    return correlations, y_data

def similarity_euclidean_preproc(embs, embs_name, dictionary, correlations={}, y_data={}, filter_dictionary=None):
    scores = {}
    if filter_dictionary is None:
        filter_dictionary = dictionary

    similarity_euclidean(center_and_normalize_eucl(embs, True, False, 0), embs_name + "-cn", dictionary,
                         correlations=correlations, y_data=y_data, filter_dictionary=filter_dictionary)
    similarity_euclidean(center_and_normalize_eucl(embs, False, True, 0), embs_name + "-nc", dictionary,
                         correlations=correlations, y_data=y_data, filter_dictionary=filter_dictionary)
    similarity_euclidean(center_and_normalize_eucl(embs, False, False, 0), embs_name + "-n", dictionary,
                         correlations=correlations, y_data=y_data, filter_dictionary=filter_dictionary)


def similarity_almost_fisher_uv(u_embeddings, v_embeddings, embs_name, dictionary, correlations={}, y_data={}, p0=None, filter_dictionary=None):
    # SIMILARITIES FISHER U V
    scores = {}
    if filter_dictionary is None:
        filter_dictionary = dictionary

    method_name = embs_name + "-cosmod"

    def simeval(words1, words2):
        return similarity_cosUVmod(u_embeddings, v_embeddings, dictionary, words1, words2, p0=p0)

    similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data)

    return correlations, y_data

def similarity_double_cosprod(u_embs, v_embs, embs_name, dictionary, correlations={}, y_data={}, filter_dictionary=None):
    # SIMILARITIES double cosprod
    scores = {}
    if filter_dictionary is None:
        filter_dictionary = dictionary

    method_name = embs_name + "-dbcosprod"

    def simeval(words1, words2):
        return similarity_cosprod(u_embs, dictionary, words1, words2) + similarity_cosprod(v_embs, dictionary, words1, words2)

    similarity_calc_and_org(method_name, simeval, filter_dictionary, scores, correlations, y_data)

    return correlations, y_data


def mean_numexpr(embs, index):
    if index==0:
        mean = ne.evaluate("sum(embs, axis=0)") / embs.shape[0]
        return mean.reshape(1, -1)
    elif index==1:
        mean = ne.evaluate("sum(embs, axis=1)") / embs.shape[1]
        return mean.reshape(-1, 1)
    else:
        raise Exception("index can be either 0 or 1")


def center_and_normalize_eucl(embs, center_before=False, center_after=True, center_index=1, normalize=True):

    if center_index not in [0, 1]:
        raise Exception("center_index can be either 0 for center columns or 1 for center rows")

    # is_mat = True
    # # I need a np.matrix for this function
    # if type(embs) is not np.matrixlib.defmatrix.matrix or type(embs) is not scipy.sparse.coo.coo_matrix:
    #     is_mat = False
    #     embs = np.matrix(embs)

    if center_before:
        # embs = embs - np.mean(embs, axis=center_index)
        embs = embs - mean_numexpr(embs, center_index)

    if normalize:
        # import pdb;pdb.set_trace()
        norms = np.sqrt(ne.evaluate("sum(embs**2, axis=0)")).reshape(1,-1)
        # norms = np.linalg.norm(embs, axis=0)
        # embs = embs / norms
        embs = ne.evaluate("embs / norms")
        # embs = embs / np.sqrt(np.sum(embs**2, axis=0))
        # embs = embs / np.linalg.norm(embs, axis=0)

    if center_after:
        # embs = embs - np.mean(embs, axis=center_index)
        embs = embs - mean_numexpr(embs, center_index)

    # if not is_mat:
    #     embs = np.array(embs)

    return embs


def merge(a, b):
    "merges b into a"
    for key in b:
        if isinstance(b[key], dict):
            if (key in a) and isinstance(a[key], dict):
                merge(a[key], b[key])
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a


def compute_p_wc_from_counts(C):
    print("I am creating p_wc from C...")
    N = C.sum()
    sums1 = C.sum(axis=1).reshape(-1,1)
    p_w = sums1 / N
    # p_c = C.sum(axis=0) / N
    p_cIw = C / sums1
    # p_cIw = ne.evaluate("C / sums1")
    # p_wc = np.multiply(p_cIw, p_w)
    p_wc = ne.evaluate("p_cIw * p_w")
    # print("I am creating the ratio matrix...")
    # prodprobs = np.matmul(p_w, p_c)
    # # r1 = p_wc / prodprobs
    # # r2 = p_cIw / p_c
    # r = (C/np.matmul(C.sum(axis=1), C.sum(axis=0))) * N

    # print("I am creating PMI, NPMI and PPMI matrices...")
    # PMI = np.log(r+NUMTOL)
    # NPMI = PMI/(-np.log(p_wc+NUMTOL))
    # PPMI = np.maximum(NUMTOL, PMI) - NUMTOL
    #
    # x_data = np.sqrt(p_cIw)
    # h_data = np.log(1+p_cIw)

    p_w = np.squeeze(np.array(p_w))
    # p_c = np.squeeze(np.array(p_c))
    # np.allclose(p_w, p_c)
    return p_w, p_wc

def compute_PMI_from_counts(C):
    print("I am creating the ratio matrix...")
    # prodprobs = np.matmul(p_w, p_c)
    # # r1 = p_wc / prodprobs
    # # r2 = p_cIw / p_c
    N = C.sum()
    r = (C/np.matmul(C.sum(axis=1), C.sum(axis=0))) * N

    print("I am creating PMI...")
    PMI = np.log(r+NUMTOL)
    # NPMI = PMI/(-np.log(p_wc+NUMTOL))
    # PPMI = np.maximum(NUMTOL, PMI) - NUMTOL
    #
    # x_data = np.sqrt(p_cIw)
    # h_data = np.log(1+p_cIw)

    # p_w = np.squeeze(np.array(p_w))
    # p_c = np.squeeze(np.array(p_c))
    # np.allclose(p_w, p_c)
    return PMI

def compute_PPMI_from_counts(C):
    print("I am creating the ratio matrix...")
    # prodprobs = np.matmul(p_w, p_c)
    # # r1 = p_wc / prodprobs
    # # r2 = p_cIw / p_c
    N = C.sum()
    r = (C / np.matmul(C.sum(axis=1), C.sum(axis=0))) * N

    print("I am creating PMI...")
    PMI = np.log(r + NUMTOL)
    # NPMI = PMI/(-np.log(p_wc+NUMTOL))
    PPMI = np.maximum(NUMTOL, PMI) - NUMTOL
    return PPMI


def compute_p_cIw_from_counts(C):
    print("I am creating pcIw from C...")
    N = C.sum()
    sums1 = C.sum(axis=1).reshape(-1,1)
    p_w = sums1 / N
    # p_c = C.sum(axis=0) / N
    p_cIw = C / sums1
    # p_wc = np.multiply(p_cIw, p_w)

    # print("I am creating the ratio matrix...")
    # prodprobs = np.matmul(p_w, p_c)
    # # r1 = p_wc / prodprobs
    # # r2 = p_cIw / p_c
    # r = (C/np.matmul(C.sum(axis=1), C.sum(axis=0))) * N

    # print("I am creating PMI, NPMI and PPMI matrices...")
    # PMI = np.log(r+NUMTOL)
    # NPMI = PMI/(-np.log(p_wc+NUMTOL))
    # PPMI = np.maximum(NUMTOL, PMI) - NUMTOL
    #
    # x_data = np.sqrt(p_cIw)
    # h_data = np.log(1+p_cIw)

    p_w = np.squeeze(np.array(p_w))
    # p_c = np.squeeze(np.array(p_c))
    # np.allclose(p_w, p_c)
    return p_w, p_cIw


def compute_probs_from_counts(C):
    print("I am creating probabilities matrices...")
    N = C.sum()
    p_w = C.sum(axis=1) / N
    p_c = C.sum(axis=0) / N
    p_cIw = C / C.sum(axis=1)
    p_wc = np.multiply(p_cIw, p_w)

    # print("I am creating the ratio matrix...")
    # prodprobs = np.matmul(p_w, p_c)
    # # r1 = p_wc / prodprobs
    # # r2 = p_cIw / p_c
    # r = (C/np.matmul(C.sum(axis=1), C.sum(axis=0))) * N

    # print("I am creating PMI, NPMI and PPMI matrices...")
    # PMI = np.log(r+NUMTOL)
    # NPMI = PMI/(-np.log(p_wc+NUMTOL))
    # PPMI = np.maximum(NUMTOL, PMI) - NUMTOL
    #
    # x_data = np.sqrt(p_cIw)
    # h_data = np.log(1+p_cIw)

    p_w = np.squeeze(np.array(p_w))
    p_c = np.squeeze(np.array(p_c))
    np.allclose(p_w, p_c)
    return p_w, p_cIw, p_wc

def calculate_or_load_common_base(fnamecc, v_dictionary, outdirname, corrs, y_data):
    json_dir = os.path.split(outdirname)[0]
    _id = os.path.splitext(os.path.basename(fnamecc))[0]
    json_name = json_dir + "/" + _id + "-common_base.json"

    try:

        with open(json_name, 'r') as fstream:
            cb_corrs, cb_y_data = json.load(fstream)

    except:

        cb_corrs = {}
        cb_y_data = {}
        C = read_cooccurrences_from_c(fnamecc)
        p_w, p_cIw, p_wc = compute_probs_from_counts(C)

        similarity_euclidean(p_wc, "p_wc", v_dictionary, correlations=cb_corrs, y_data=cb_y_data)
        similarity_euclidean(p_cIw, "p_cIw", v_dictionary, correlations=cb_corrs, y_data=cb_y_data)
        del p_wc, p_cIw, p_w
        gc.collect()

        g_dict, g_vecs = load_pretrained_glove("wikigiga5")
        similarity_euclidean(g_vecs["u"], "wikigiga5-u+v", g_dict,
                             correlations=cb_corrs, y_data=cb_y_data, filter_dictionary=v_dictionary)

        # # COMMON CRAWL
        # g_dict, g_vecs = load_pretrained_glove("commoncrawl42B")
        # similarity_euclidean(g_vecs["u"], "commoncrawl42B-u+v", g_dict,
        #                      correlations=cb_corrs, y_data=cb_y_data, filter_dictionary=v_dictionary)
        #
        # g_dict, g_vecs = load_pretrained_glove("commoncrawl840B")
        # keys_a = set(g_dict.keys())
        # keys_b = set(v_dictionary.keys())
        # intersection_keys = keys_a & keys_b
        # similarity_euclidean(g_vecs["u"], "commoncrawl840B-u+v", g_dict,
        #                      correlations=cb_corrs, y_data=cb_y_data, filter_dictionary=intersection_keys)

    with open(json_name, 'w') as fstream:
            json.dump([cb_corrs, cb_y_data], fstream)

    merge(corrs, cb_corrs)
    merge(y_data, cb_y_data)


def calculate_or_load_common_base_preproc(fnamecc, v_dictionary, outdirname, corrs, y_data):
    json_dir = os.path.split(outdirname)[0]
    _id = os.path.basename(fnamecc)
    json_name = json_dir + "/"+_id+"-common_base_preproc.json"

    try:
        with open(json_name, 'r') as fstream:
            cb_corrs, cb_y_data = json.load(fstream)

    except:

        cb_corrs = {}
        cb_y_data = {}

        C = read_cooccurrences_from_c(fnamecc)

        # p_w, p_wc = compute_p_wc_from_counts(C)
        #
        # times.append(time.time())
        #
        # corrs = {}
        # y_data = {}
        #
        # similarity_euclidean_preproc(p_wc, "p_wc", v_dictionary, correlations=corrs, y_data=y_data)
        #
        # times.append(time.time())
        #
        # del p_wc
        # gc.collect()

        p_w, p_cIw = compute_p_cIw_from_counts(C)

        similarity_euclidean_preproc(p_cIw, "p_cIw", v_dictionary, correlations=cb_corrs, y_data=cb_y_data)
        del p_cIw
        gc.collect()

        g_dict, g_vecs = load_pretrained_glove("wikigiga5")
        similarity_euclidean_preproc(g_vecs["u"], "wikigiga5-u+v", g_dict,
                                     correlations=cb_corrs, y_data=cb_y_data, filter_dictionary=v_dictionary)

        # # COMMON CRAWL
        # g_dict, g_vecs = load_pretrained_glove("commoncrawl42B")
        # similarity_euclidean_preproc(g_vecs["u"], "commoncrawl42B-u+v", g_dict,
        #                      correlations=cb_corrs, y_data=cb_y_data, filter_dictionary=v_dictionary)
        #
        # g_dict, g_vecs = load_pretrained_glove("commoncrawl840B")
        # keys_a = set(g_dict.keys())
        # keys_b = set(v_dictionary.keys())
        # intersection_keys = keys_a & keys_b
        # similarity_euclidean_preproc(g_vecs["u"], "commoncrawl840B-u+v", g_dict,
        #                      correlations=cb_corrs, y_data=cb_y_data, filter_dictionary=intersection_keys)

        with open(json_name, 'w') as fstream:
            json.dump([cb_corrs, cb_y_data], fstream)

    merge(corrs, cb_corrs)
    merge(y_data, cb_y_data)


def base_similarities(corpus, vecsize, nepoch, fnamecc, v_dictionary, ref_keys_no_preproc, ref_keys_preproc, outdirname):
    suffix = get_suffix(vecsize, nepoch)
    base_json_name = outdirname+"/base_data_ref"+suffix+".json"
    basep_json_name = outdirname+"/base_preproc_data_ref"+suffix+".json"

    try:
        with open(base_json_name, 'r') as fstream:
            y_data_ref_nop = json.load(fstream)

        with open(basep_json_name, 'r') as fstream:
            y_data_ref_p = json.load(fstream)

    except:

        corrs={}
        y_data = {}
        calculate_or_load_common_base(fnamecc, v_dictionary, outdirname, corrs, y_data)

        g_dict, g_vecs, _ = load_glove(corpus, vecsize, nepoch, calc_prob=False)
        similarity_euclidean(g_vecs["u"], corpus + "-u", v_dictionary, correlations=corrs, y_data=y_data)
        similarity_euclidean(g_vecs["v"], corpus + "-v", v_dictionary, correlations=corrs, y_data=y_data)
        similarity_euclidean(g_vecs["u"] + g_vecs["v"], corpus + "-u+v", v_dictionary, correlations=corrs, y_data=y_data)
        similarity_double_cosprod(g_vecs["u"], g_vecs["v"], corpus + "-uvDcos", v_dictionary, correlations=corrs,
                                  y_data=y_data)

        similarity_fisher_uv(g_vecs["u"], g_vecs["v"], corpus, v_dictionary, correlations=corrs, y_data=y_data)
        # similarity_fisher_uv(g_vecs["u"], project_away_1vec_component(g_vecs["v"]), gmn+"-o1", v_dictionary, correlations=corrs, y_data=y_data)

        # similarity_almost_fisher_uv(g_vecs["u"], g_vecs["v"], gmn + "-0-uv", v_dictionary, correlations=corrs, y_data=y_data)
        # similarity_almost_fisher_uv(g_vecs["u"], g_vecs["v"], gmn + "-u-uv", v_dictionary,
        #                      correlations=corrs, y_data=y_data, p0=p_w)

        print_table(corrs, outdirname+"/base"+suffix+".txt")

        # corrs_ref = {k:v for k, v in corrs.items() if k in ref_keys_no_preproc}
        y_data_ref_nop = {}
        for task_key, task_dict in y_data.items():
            task_ref_dict = {k: v for k, v in task_dict.items() if k in ref_keys_no_preproc}
            y_data_ref_nop[task_key] = task_ref_dict


        print("I start with preprocessing")

        # PREPROCESSING

        # center and normalize columns of p_wc and p_cIw before cosprod
        # center and normalize columns of U and V before cosprod

        # center and normalize columns of counts before computing p_cIw, then logmap
        # center and normalize columns of U and V before computing p, then logmap


        calculate_or_load_common_base_preproc(fnamecc, v_dictionary, outdirname, corrs, y_data)


        g_dict, g_vecs, _ = load_glove(corpus, vecsize, nepoch, calc_prob=False)

        similarity_euclidean_preproc(g_vecs["u"], corpus + "-u", v_dictionary,
                                     correlations=corrs, y_data=y_data)

        similarity_euclidean_preproc(g_vecs["v"], corpus + "-v", v_dictionary,
                                     correlations=corrs, y_data=y_data)

        similarity_euclidean_preproc(g_vecs["u"] + g_vecs["v"], corpus + "-u+v",
                                     v_dictionary, correlations=corrs, y_data=y_data)

        similarity_double_cosprod(center_and_normalize_eucl(g_vecs["u"], True, False, 0),
                                  center_and_normalize_eucl(g_vecs["v"], True, False, 0), corpus + "-uvDcos-cn",
                                  v_dictionary, correlations=corrs, y_data=y_data)

        print_table(corrs, outdirname+"/base-preproc"+suffix+".txt")

        # corrs_ref = {k:v for k, v in corrs.items() if k in ref_keys_preproc}
        y_data_ref_p = {}
        for task_key, task_dict in y_data.items():
            task_ref_dict = {k: v for k, v in task_dict.items() if k in ref_keys_preproc}
            y_data_ref_p[task_key] = task_ref_dict

        for task_key, task_dict in y_data.items():
            task_ref_dict = {k: v for k, v in task_dict.items() if k in ref_keys_no_preproc}

            if y_data_ref_nop.get(task_key, None) is None:
                y_data_ref_nop[task_key] = {}
            y_data_ref_nop[task_key].update(task_ref_dict)

        with open(base_json_name, 'w') as fstream:
            json.dump(y_data_ref_nop, fstream)

        with open(basep_json_name, 'w') as fstream:
            json.dump(y_data_ref_p, fstream)


    return y_data_ref_nop, y_data_ref_p



def similarities_u_scale(U, V, scale, alphas, gname, v_dictionary, outdirname, y_data_ref):
    corrs = {}
    y_data = {}
    U_mult = scale * U
    V_mult = V
    g_p_w, g_p_cIw = calculate_glove_prob(U_mult, V_mult)

    similarities_logmaps_Esubmodel_trick(g_p_cIw, U_mult, V_mult, "p_cIw-m-E", g_p_w, alphas, v_dictionary, corrs, y_data, method="cos")
    # similarities_logmaps_Esubmodel(g_p_cIw, U, project_away_1vec_component(V), "p_cIw-o1", g_p_w, alphas,
    #                                v_dictionary, corrs, y_data)
    output = outdirname+"/logmaps-p_" + gname + "-Esubmodel-cos-uscale%s"%str(scale)
    print_table(corrs, output + ".txt")
    merge(y_data, y_data_ref)
    plot_all_tasks(alphas, y_data, output)



def all_log_similarities(corpus, vecsize, nepoch, alphas, v_dictionary, outdirname, y_data_ref={}):
    # # FULL SIMPLEX FROM DATA COMMENT
    # C = read_cooccurrences_from_c(fnamecc)
    # p_w, p_cIw, p_wc = compute_probs_from_counts(C)
    #
    # # corrs = {}
    # # y_data = {}
    # # similarities_divergences(p_cIw, "p_cIw", alphas, corrs, y_data)
    # # print_table(corrs, "divergence-p_data.txt")
    # # plot_all_tasks(alphas, y_data, "divergence-p_data")
    #
    # corrs = {}
    # y_data = {}
    #
    # similarities_logmaps_fullrank(p_cIw, "p_cIw-d-P", p_w, alphas, v_dictionary, corrs, y_data)
    # print_table(corrs, "logmaps-p_data.txt")
    # merge(y_data, y_data_ref)
    # plot_all_tasks(alphas, y_data, "logmaps-p_data")
    #
    # del C, p_wc, p_cIw, p_w
    # gc.collect()
    # # FULL SIMPLEX FROM DATA COMMENT

    gname = corpus+get_suffix(vecsize, nepoch)
    g_dict, g_vecs, g_tuple = load_glove(corpus, vecsize, nepoch, calc_prob=False)
    # g_p_w, g_p_cIw = g_tuple
    U = g_vecs["u"][:-1, :]
    V = g_vecs["v"][:-1, :]

    # #MODEL FULL RANK COMMENT
    # corrs = {}
    # y_data = {}
    # similarities_logmaps_fullrank(g_p_cIw, "p_cIw-m-P", g_p_w, alphas, v_dictionary, corrs, y_data)
    # print_table(corrs, "logmaps-p_" + gmn + "-full.txt")
    # merge(y_data, y_data_ref)
    # plot_all_tasks(alphas, y_data, "logmaps-p_" + gmn + "-full")
    # #MODEL FULL RANK COMMENT

    # corrs = {}
    # y_data = {}
    # similarities_logmaps_Esubmodel(g_p_cIw, U, V, "p_cIw-m-E", g_p_w, alphas, v_dictionary, corrs, y_data, method="cos")
    # # similarities_logmaps_Esubmodel(g_p_cIw, U, project_away_1vec_component(V), "p_cIw-o1", g_p_w, alphas,
    # #                                v_dictionary, corrs, y_data)
    # print_table(corrs, "logmaps-p_" + gmn + "-Esubmodel-cos.txt")
    # merge(y_data, y_data_ref)
    # plot_all_tasks(alphas, y_data, "logmaps-p_" + gmn + "-Esubmodel-cos")

    for scale in [1]: #[0.5, 1]:
        similarities_u_scale(U, V, scale, alphas, gname, v_dictionary, outdirname, y_data_ref)

    # #SUB MODEL PROJ DIST COMMENT
    # corrs = {}
    # y_data = {}
    # similarities_logmaps_Esubmodel(g_p_cIw, U, V, "p_cIw-m-E", g_p_w, alphas, v_dictionary, corrs, y_data, method="dis")
    # print_table(corrs, "logmaps-p_" + gmn + "-Esubmodel-dis.txt")
    # merge(y_data, y_data_ref)
    # plot_all_tasks(alphas, y_data, "logmaps-p_" + gmn + "-Esubmodel-dis")
    # #SUB MODEL PROJ DIST COMMENT



def similarities_u_preproc(U, V, scale, alphas, gname, v_dictionary, outdirname, y_data_ref):
    corrs = {}
    y_data = {}
    scale0 = np.mean(np.linalg.norm(U, axis=0))
    U_preproc = scale * scale0 * center_and_normalize_eucl(U, True, False, 0, normalize=True)
    V_preproc = V
    # np.testing.assert_array_equal(U, g_vecs["u"][:-1, :])
    g_p_w, g_p_cIw = calculate_glove_prob(U_preproc, V_preproc)
    similarities_logmaps_Esubmodel(g_p_cIw, U_preproc, V_preproc, "p_cIw-ucn-m-E", g_p_w, alphas,
                                   v_dictionary, corrs, y_data, method="cos")

    output = outdirname+"/logmaps-p_" + gname + "-Esubmodel-preproc-u-n%.2f-scale%s"%(scale0, str(scale))
    print_table(corrs, output +".txt")
    merge(y_data, y_data_ref)
    plot_all_tasks(alphas, y_data, output)

    # corrs = {}
    # y_data = {}
    # similarities_logmaps_fullrank(g_p_cIw, "p_cIw-ucn-m-P", g_p_w, alphas, v_dictionary, corrs, y_data)
    # print_table(corrs, "logmaps-p_" + gmn + "-full-preproc-u-scale%s.txt"%("%.2f"%scale))
    # merge(y_data, y_data_ref)
    # plot_all_tasks(alphas, y_data, "logmaps-p_" + gmn + "-full-preproc-u-scale%s"%("%.2f"%scale))




# @profile
def all_log_similarities_preproc(corpus, vecsize, nepoch, alphas, v_dictionary, outdirname, y_data_ref={}):
    # ##COMMENTED
    # C = read_cooccurrences_from_c(fnamecc)
    #
    # corrs = {}
    # y_data = {}
    #
    # C = C / C.sum(axis=0).reshape(1,-1)
    # p_w, p_cIw = compute_p_cIw_from_counts(C)
    # similarities_logmaps_fullrank(p_cIw, "p_cIw-Cds-d-P", p_w, alphas, v_dictionary, corrs, y_data)
    #
    # del C, p_w, p_cIw
    # gc.collect()
    #
    # # C = read_cooccurrences_from_c(fnamecc)
    # # C = C / np.sqrt(np.sum(C ** 2, axis=0)).reshape(1, -1)
    # # C = C / np.linalg.norm(C, axis=0).reshape(1, -1)
    # # p_w, p_cIw = compute_p_cIw_from_counts(C)
    # # similarities_logmaps_fullrank(p_cIw, "p_cIw-Cn", p_w, alphas, v_dictionary, corrs, y_data)
    # # del p_cIw, C, p_w
    # # gc.collect()
    #
    # # C_proc = C / np.amax(C, axis=0)
    # # N, p_w, p_cIw, p_wc = compute_probs_from_counts(C_proc)
    # # similarities_logmaps_fullrank(p_cIw, "p_cIw-dm", alphas, corrs, y_data)
    # # similarities_logmaps_fullrank(p_wc, "p_wc-dm", alphas, corrs, y_data)
    #
    # print_table(corrs, "logmaps-p_data-preproc.txt")
    # merge(y_data, y_data_ref)
    # plot_all_tasks(alphas, y_data, "logmaps-p_data-preproc")

    gname = corpus+get_suffix(vecsize, nepoch)
    g_dict, g_vecs, g_tuple = load_glove(corpus, vecsize, nepoch, calc_prob=False)

    U = g_vecs["u"][:-1, :]
    V = g_vecs["v"][:-1, :]

    # OLD PREPROC on the model
    # g_p_w, g_p_cIw = calculate_glove_prob(
    #             center_and_normalize_eucl(g_vecs["u"], True, False, 0, normalize=False),
    #             center_and_normalize_eucl(g_vecs["v"], True, False, 0, normalize=False))
    #
    # # p_w = g_prob.sum(axis=1)
    # similarities_logmaps_fullrank(g_p_cIw, "p_cIw-uvc", g_p_w, alphas, v_dictionary, corrs, y_data)
    #
    # # except:
    # #     pdb.set_trace()
    #
    # g_p_w, g_p_cIw = calculate_glove_prob(
    #                     g_vecs["u"],
    #                     center_and_normalize_eucl(g_vecs["v"], True, False, 0, normalize=False))
    #
    # similarities_logmaps_fullrank(g_p_cIw, "p_cIw-vc", g_p_w, alphas, v_dictionary, corrs, y_data)
    #
    # g_p_w, g_p_cIw = calculate_glove_prob(g_vecs["u"], g_vecs["v"], norm_counts_cols = True)
    # similarities_logmaps_fullrank(g_p_cIw, "p_cIw-Cn", g_p_w, alphas, v_dictionary, corrs, y_data)
    #
    # xi = np.matmul(g_vecs["u"], np.transpose(g_vecs["v"]))
    # similarity_euclidean(xi, "xi", v_dictionary, corrs, y_data)
    # OLD PREPROC on the model

    # you can check this projection does not change the distribution "much" np.allclose
    # g_p_w, g_p_cIw = calculate_glove_prob(g_vecs["u"], g_vecs["v"])
    # g_p_w, g_p_cIw = calculate_glove_prob(g_vecs["u"], project_away_1vec_component(g_vecs["v"]))

    #U PREPROC
    for scale in [10]:
        similarities_u_preproc(U, V, scale, alphas, gname, v_dictionary, outdirname, y_data_ref)
    #U PREPROC

    # # V PREPROC COMMENT
    # corrs = {}
    # y_data = {}
    # U_preproc = U
    # V_preproc = center_and_normalize_eucl(V, True, False, 0, normalize=True)
    # np.testing.assert_array_equal(V, g_vecs["v"][:-1, :])
    #
    # g_p_w, g_p_cIw = calculate_glove_prob(U_preproc, V_preproc)
    # similarities_logmaps_Esubmodel(g_p_cIw, U_preproc, V_preproc, "p_cIw-vcn-m-E", g_p_w, alphas,
    #                                v_dictionary, corrs, y_data, method="cos")
    #
    # print_table(corrs, "logmaps-p_"+gmn+"-Esubmodel-preproc-v.txt")
    # merge(y_data, y_data_ref)
    # plot_all_tasks(alphas, y_data, "logmaps-p_"+gmn+"-Esubmodel-preproc-v")
    #
    # corrs = {}
    # y_data = {}
    # similarities_logmaps_fullrank(g_p_cIw, "p_cIw-ucn-m-P", g_p_w, alphas, v_dictionary, corrs, y_data)
    # print_table(corrs, "logmaps-p_" + gmn + "-full-preproc-v.txt")
    # merge(y_data, y_data_ref)
    # plot_all_tasks(alphas, y_data, "logmaps-p_" + gmn + "-full-preproc-v")
    # # V PREPROC COMMENT

    # # UV PREPROC
    #
    # corrs = {}
    # y_data = {}
    # U_preproc = center_and_normalize_eucl(U, True, False, 0, normalize=True)
    # V_preproc = center_and_normalize_eucl(V, True, False, 0, normalize=True)
    # np.testing.assert_array_equal(U, g_vecs["u"][:-1, :])
    # np.testing.assert_array_equal(V, g_vecs["v"][:-1, :])
    #
    # g_p_w, g_p_cIw = calculate_glove_prob(U_preproc, V_preproc)
    # similarities_logmaps_Esubmodel(g_p_cIw, U_preproc, V_preproc, "p_cIw-uvcn-m-E", g_p_w, alphas,
    #                                v_dictionary, corrs, y_data, method="cos")
    #
    # print_table(corrs, "logmaps-p_"+gmn+"-Esubmodel-preproc-uv.txt")
    # merge(y_data, y_data_ref)
    # plot_all_tasks(alphas, y_data, "logmaps-p_"+gmn+"-Esubmodel-preproc-uv")
    #
    # corrs = {}
    # y_data = {}
    # similarities_logmaps_fullrank(g_p_cIw, "p_cIw-uvcn-m-P", g_p_w, alphas, v_dictionary, corrs, y_data)
    # print_table(corrs, "logmaps-p_" + gmn + "-full-preproc-uv.txt")
    # merge(y_data, y_data_ref)
    # plot_all_tasks(alphas, y_data, "logmaps-p_" + gmn + "-full-preproc-uv")
    #


# @profile
def test_sparse_matrices(C):
    import time

    times = []
    times.append(time.time())
    C_csc = scipy.sparse.csc_matrix(C)
    times.append(time.time())
    C1_csc = C_csc / C_csc.sum(axis=0).reshape(1,-1)
    times.append(time.time())
    C2_csc = C_csc.mean(axis=0)
    times.append(time.time())
    C3_csc = C_csc.mean(axis=1)
    times.append(time.time())
    deltas_csc = np.array(times[1:]) - np.array(times[:-1])

    times = []
    times.append(time.time())
    C_csr = scipy.sparse.csr_matrix(C)
    times.append(time.time())
    C1_csr = C_csr / C_csr.sum(axis=0).reshape(1,-1)
    times.append(time.time())
    C2_csr = C_csr.mean(axis=0)
    times.append(time.time())
    C3_csr = C_csr.mean(axis=1)
    times.append(time.time())
    deltas_csr = np.array(times[1:]) - np.array(times[:-1])

    print("csr:", deltas_csr)
    print("csc:", deltas_csc)

    print(np.allclose(C1_csr, C1_csc))
    print(np.allclose(C2_csr, C2_csc))
    print(np.allclose(C3_csr, C3_csc))

def test_huge_sparse_matrices(C):
    import time

    times = []
    times.append(time.time())
    C_csc = scipy.sparse.csc_matrix(C)
    times.append(time.time())
    C_csc / C_csc.sum(axis=0).reshape(1,-1)
    times.append(time.time())
    C_csc.mean(axis=0)
    times.append(time.time())
    C_csc.mean(axis=1)
    times.append(time.time())
    deltas_csc = np.array(times[1:]) - np.array(times[:-1])

    del C_csc
    gc.collect()

    times = []
    times.append(time.time())
    C_csr = scipy.sparse.csr_matrix(C)
    times.append(time.time())
    C_csr / C_csr.sum(axis=0).reshape(1,-1)
    times.append(time.time())
    C_csr.mean(axis=0)
    times.append(time.time())
    C_csr.mean(axis=1)
    times.append(time.time())
    deltas_csr = np.array(times[1:]) - np.array(times[:-1])

    del C_csr
    gc.collect()
    
    print("csr:", deltas_csr)
    print("csc:", deltas_csc)


def plot_isotropicity(glove_models_names, outdirname):
    # isotropicity of v vectors
    all_eigs = {}
    for gmn in glove_models_names:
        _, g_vecs, _ = load_glove(gmn, calc_prob=False)
        X = np.transpose(g_vecs["v"][:-1, :])
        D = X.shape[1]
        Cov = np.matmul(X, X.T) / (D - 1)
        eigs, V = scipy.sparse.linalg.eigsh(Cov)
        all_eigs[gmn] = eigs

        X = np.transpose(center_and_normalize_eucl(g_vecs["v"][:-1, :], True, False, 0))
        # X = X - X.mean(axis=1).reshape(-1,1)
        # X = X / np.linalg.norm(X, axis=1).reshape(-1,1)
        Cov = np.matmul(X, X.T) / (D - 1)
        eigs, V = scipy.sparse.linalg.eigsh(Cov)
        all_eigs[gmn+"-cn"] = eigs

        X = np.transpose(center_and_normalize_eucl(g_vecs["v"][:-1, :], False, False, 0))
        # X = X - X.mean(axis=1).reshape(-1,1)
        # X = X / np.linalg.norm(X, axis=1).reshape(-1,1)
        Cov = np.matmul(X, X.T) / (D - 1)
        eigs, V = scipy.sparse.linalg.eigsh(Cov)
        all_eigs[gmn+"-c"] = eigs

    for gmn in all_eigs:
        eigs = np.array(sorted(all_eigs[gmn], reverse=True))
        print(gmn)
        print(eigs)
        eigs = eigs[1:]
        plt.plot(eigs/eigs[0], label=gmn)

    plt.legend()
    plt.savefig(outdirname+"/isotropy_v.png")


def make_all_sims(corpus, vecsize, nepoch, alphas, baseoutdir, exp_name_for_dir):

    suffix = get_suffix(vecsize, nepoch)

    outdirname = os.path.join(baseoutdir, corpus + exp_name_for_dir + "/" + corpus + suffix)

    os.makedirs(outdirname, exist_ok=True)

    dsdir = '/ssd_data/text/cooccurrences/'
    # simplewiki_sw6_fnamecc = dsdir + 'simplewiki201711/simplewiki201711-sw6-cooccurrence.bin'
    simplewiki_sw10_fnamecc = dsdir + 'simplewiki201711/simplewiki201711-sw10-cooccurrence.bin'
    simplewiki_fnamevc = dsdir + 'simplewiki201711/simplewiki201711-vocab.txt'

    enwiki_sw10_fnamecc = dsdir + 'enwiki201710/enwiki201710-sw10-cooccurrence.bin'
    enwiki_fnamevc = dsdir + 'enwiki201710/enwiki201710-vocab.txt'

    # select which vocabulary and cooccurrence file to use
    if corpus == "enwiki":
        fnamevc = enwiki_fnamevc
        fnamecc = enwiki_sw10_fnamecc
        ref_keys_no_preproc = ["enwiki-u+v-n-cosprod", "enwiki-u-cosprod", "wikigiga5-u+v-n-cosprod",
                            "p_cIw-cn-cosprod"]
        ref_keys_preproc = ["enwiki-u+v-n-cosprod", "enwiki-u-n-cosprod", "wikigiga5-u+v-n-cosprod", "p_cIw-cn-cosprod"] #, "wikigiga5-u+v-c-cosprod"]

    elif corpus == "swiki":
        fnamevc = simplewiki_fnamevc
        fnamecc = simplewiki_sw10_fnamecc
        # glove_models_names = ["swiki-500", "swiki-1000"]
        # glove_models_names = ["swiki-1000"]
        ref_keys_no_preproc = ["swiki-u+v-n-cosprod", "swiki-u-cosprod", "wikigiga5-u+v-n-cosprod", "p_cIw-cn-cosprod"]  # , "wikigiga5-u+v-c-cosprod"]
        ref_keys_preproc = ["swiki-u+v-n-cosprod", "swiki-u-n-cosprod", "wikigiga5-u+v-n-cosprod", "p_cIw-cn-cosprod"] #, "wikigiga5-u+v-c-cosprod"]

    else:
        raise ValueError("corpus not recognized `%s`" % corpus)

    # w2v_reader = readers.get_reader("word2vec")

    (dictionary_size, v_dictionary, v_reversed_dictionary) = g_reader.read_dictionary(fnamevc)

    # plot_isotropicity()

    # C = read_cooccurrences_from_c(fnamecc)
    # test_sparse_matrices(C)
    # sys.exit(0)
    with open(outdirname+"/alphas.json", 'w') as fstream:
        json.dump({"alphas" : list(alphas)}, fstream)

    y_data_ref_nop, y_data_ref_p = base_similarities(corpus, vecsize, nepoch, fnamecc, v_dictionary,
                                                     ref_keys_no_preproc, ref_keys_preproc, outdirname)
    # y_data_ref={}
    all_log_similarities(corpus, vecsize, nepoch, alphas, v_dictionary, outdirname, y_data_ref=y_data_ref_nop)
    # all_log_similarities_preproc(corpus, vecsize, nepoch, alphas, v_dictionary, outdirname, y_data_ref=y_data_ref_p)


g_reader = readers.get_reader("glove")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Similarities logmaps and alphas.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('corpus', choices=["swiki", "enwiki"], help='Corpus for which to evaluate similarities.')
    parser.add_argument('--exp', required=True, help='small, mid or large. Define the alpha range for the experiment.')
    parser.add_argument('--outputdir', '-o', help='outputdir')


    # parser.add_argument('vecsize', type=int, help='the size of glove to load.')
    # parser.add_argument('epoch', type=int, help='the training epoch of glove to load.')
    # glove_vecsize = args.vecsize
    # glove_nepoch = args.epoch

    args = parser.parse_args()
    corpus = args.corpus
    # CORPUS = "swiki"
    # CORPUS = "enwiki"

    exp_name = args.exp
    if exp_name == "small":
        exp_name_for_dir = "-small-alphas"
        alphas = np.arange(-5, 5, 0.1)

    elif exp_name == "mid":
        exp_name_for_dir = "-mid-alphas"
        alphas = np.arange(-20., 22., 2.)

    elif exp_name == "large":
        exp_name_for_dir = "-large-alphas"
        alphas = np.arange(-50., 60., 10.)

    else:
        raise ValueError("exp_name not valid")

    #old was -> "/data1/text/similarities/results/"
    baseoutdir = args.outputdir

    for vecsize in [300]: #[100, 200, 300, 400]:
        for nepoch in [1000]: #[200, 400, 600, 800, 1000]:
            make_all_sims(corpus, vecsize, nepoch, alphas, baseoutdir, exp_name_for_dir)



