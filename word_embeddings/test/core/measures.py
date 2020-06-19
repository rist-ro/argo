import glob, os, itertools
import csv
import numpy as np
from scipy.stats import spearmanr
from matplotlib import pyplot as plt
from .plotting import create_plot_data_dict, initialize_plot, plot_data, finalize_plot, save_object
import numexpr as ne

import pdb

def evaluate_similarities(dictionary, W, g, logger, filter_dictionary=None, datasets=[], outbasename=None, plot=False):
    """Evaluate the trained word vectors on a variety of similarity files
    """

    if filter_dictionary is None:
        filter_dictionary = dictionary

    folder = '/data2/text/similarities_datasets/splitted'

    if not datasets:
        # in each of these folders I expect to find the splits
        datasets = [
                "wordsim353", "mc", "rg", "scws",
                "wordsim353sim", "wordsim353rel",
                "men", "mturk287", "rw", "simlex999"
                ]

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
            ind1, ind2, humscores = zip(
                *((dictionary[w1], dictionary[w2], sc) for (w1, w2, sc) in lines if (w1 in filter_dictionary) and (w2 in filter_dictionary)))
            count_tot += len(ind1)

        #     full_data = [line.rstrip().split(' ') for line in f]
        #     full_count += len(full_data)
        #     data = [x for x in full_data if all(word in dictionary for word in x)]
        #
        # indices = np.array([[dictionary[word] for word in row] for row in data])
        # ind1, ind2, ind3, ind4 = indices.T
        # read csv in a table and then calculate the distances and pair them with the scores from the csv
        humscores = np.array(humscores, dtype=np.float)
        simscores = riemannian_cosprod(W, g, ind1, ind2)

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


        scores_dict[label] = (humscores, simscores)

        corr, p_value = spearmanr(humscores, simscores)
        corr = corr * 100
        corr_dict[label] = corr

        logger.info("%s:" % filenames[i])
        logger.info('SPEARMAN CORR: %.2f ' % corr)

        all_humscores += list(normalize(humscores, min(humscores), max(humscores)))
        all_simscores += list(simscores)

    # aggregate scores for the full datasets
    aggr_humscores = {}
    aggr_simscores = {}
    for d in datasets:
        aggr_humscores[d] = []
        aggr_simscores[d] = []

    for task, (hums, sims) in scores_dict.items():
        d = task.split("-")[0]
        aggr_humscores[d].append(hums)
        aggr_simscores[d].append(sims)

    #store aggregated scores and corrs
    for d in datasets:
        hums = np.concatenate(aggr_humscores[d])
        sims = np.concatenate(aggr_simscores[d])
        scores_dict[d] = (hums, sims)
        corr, p_value = spearmanr(hums, sims)
        corr = corr * 100
        corr_dict[d] = corr

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


def normalize(tensor, min_orig, max_orig, min_out=0, max_out=1.):
    delta = max_out - min_out
    return delta * (tensor - min_orig) / (max_orig - min_orig) + min_out

def evaluate_similarity_on_reverse_split(filter_dictionary, simeval, dataset, i_split):

    folder='/data2/text/similarities_datasets/splitted'

    split_part = "split_{:d}".format(i_split)
    label = dataset+"-"+split_part
    dirname = dataset
    fsplits = glob.glob(os.path.join(folder, os.path.join(dirname, "*.csv")))

    # all the splits except the one corresponding to index i
    fsplits = [f for f in fsplits if not (split_part in f)]

    #now merge all the fsplits
    lines = []
    for fname in fsplits:
        with open(fname, 'r') as instream:
            reader = csv.reader(instream, delimiter=' ', skipinitialspace=True)
            lines += list(reader)


    # full_count = len(lines)
    ind1, ind2, humscores = zip(
        *((filter_dictionary[w1], filter_dictionary[w2], sc) for (w1, w2, sc) in lines if (w1 in filter_dictionary) and (w2 in filter_dictionary)))

    # count_tot = len(words1)

    humscores = np.array(humscores, dtype=np.float)
    simscores = simeval(ind1, ind2)

    corr, p_value = spearmanr(humscores, simscores)
    corr = corr * 100
    return corr


def euclidean_cosprod(embeddings, ind1, ind2, normalize=True):
    emb1 = embeddings[np.asarray(ind1)]
    emb2 = embeddings[np.asarray(ind2)]

    scalprods = np.sum(emb1 * emb2, axis=1)
    if normalize:
        scalprods = scalprods / (np.linalg.norm(emb1, axis=1) * np.linalg.norm(emb2, axis=1))
    return scalprods

    # predictions = np.zeros((len(indices),))
    # num_iter = int(np.ceil(len(indices) / float(split_size)))
    # for j in range(num_iter):
    #     subset = np.arange(j * split_size, min((j + 1) * split_size, len(ind1)))
    #
    #     pred_vec = (W[ind2[subset], :] - W[ind1[subset], :]
    #                 + W[ind3[subset], :])
    #     # cosine similarity if input W has been normalized
    #     dist = np.dot(W, pred_vec.T)
    #
    #     for k in range(len(subset)):
    #         dist[ind1[subset[k]], k] = -np.Inf
    #         dist[ind2[subset[k]], k] = -np.Inf
    #         dist[ind3[subset[k]], k] = -np.Inf
    #
    #     # predicted word index
    #     predictions[subset] = np.argmax(dist, 0).flatten()
    #


def riemannian_cosprod(embeddings, I, ind1, ind2, normalize=True):
    Ca = embeddings[np.asarray(ind1)]
    Cb = embeddings[np.asarray(ind2)]

    #a is a list of vectors
    #b is a list of vectors
    #I is the metric

    np.testing.assert_array_equal([len(Ca.shape), len(Cb.shape)], [2,2])
    needed_I_shape = [Ca.shape[1], Cb.shape[1]]
    np.testing.assert_array_equal(needed_I_shape, I.shape)

    ICb = np.matmul(I, Cb.T).T
    scalprods = np.sum(Ca*ICb, axis=1)

    if normalize:
        ICa = np.matmul(I, Ca.T).T
        norms1 = np.sqrt(np.sum(Ca*ICa, axis=1))
        norms2 = np.sqrt(np.sum(Cb*ICb, axis=1))
        scalprods = scalprods / (norms1*norms2)

    return scalprods


# def evaluate_analogies_withoutsplits(dictionary, W, g, logger, filter_dictionary):
#     """Evaluate the trained word vectors on a variety of tasks"""
#
#     filenames = [
#         'capital-common-countries.txt', 'capital-world.txt', 'currency.txt',
#         'city-in-state.txt', 'family.txt', 'gram1-adjective-to-adverb.txt',
#         'gram2-opposite.txt', 'gram3-comparative.txt', 'gram4-superlative.txt',
#         'gram5-present-participle.txt', 'gram6-nationality-adjective.txt',
#         'gram7-past-tense.txt', 'gram8-plural.txt', 'gram9-plural-verbs.txt',
#     ]
#
#     folder = '/data2/text/analogies_datasets'
#
#     # to avoid memory overflow, could be increased/decreased
#     # depending on system and vocab size
#     split_size = 100
#
#     correct_sem = 0  # count correct semantic questions
#     correct_syn = 0  # count correct syntactic questions
#     correct_tot = 0  # count correct questions
#     count_sem = 0  # count all semantic questions
#     count_syn = 0  # count all syntactic questions
#     count_tot = 0  # count all questions
#     full_count = 0  # count all questions, including those with unknown words
#
#     Cq = W
#     ICq = np.matmul(g, Cq.T).T
#     sqnormq = np.sum(Cq * ICq, axis=1)
#
#     for i in range(len(filenames)):
#         with open('%s/%s' % (folder, filenames[i]), 'r') as f:
#             full_data = [line.rstrip().split(' ') for line in f]
#             full_count += len(full_data)
#             data = [x for x in full_data if all(word in filter_dictionary for word in x)]
#
#         indices = np.array([[dictionary[word] for word in row] for row in data])
#         ind1, ind2, ind3, ind4 = indices.T
#
#         howmany = 1
#         predictions = analogies_riemannian(W, g, ind1, ind2, ind3, ICq, sqnormq, howmany)
#
#         # val = (ind4 == predictions)  # correct predictions
#         val = [int(i4 in preds) for (i4, preds) in zip(ind4, predictions)]  # correct predictions
#
#         count_tot = count_tot + len(ind1)
#         correct_tot = correct_tot + sum(val)
#
#         if i < 5:
#             count_sem = count_sem + len(ind1)
#             correct_sem = correct_sem + sum(val)
#         else:
#             count_syn = count_syn + len(ind1)
#             correct_syn = correct_syn + sum(val)
#
#         logger.info("%s:" % filenames[i])
#         logger.info('ACCURACY TOP1: %.2f%% (%d/%d)' %
#               (np.mean(val) * 100, np.sum(val), len(val)))
#
#     seen_frac = 100 * count_tot / float(full_count)
#     tot_acc = 100 * correct_tot / float(count_tot)
#     sem_acc = 100 * correct_sem / float(count_sem)
#     syn_acc = 100 * correct_syn / float(count_syn)
#
#     logger.info('Questions seen/total: %.2f%% (%d/%d)' %
#           (seen_frac, count_tot, full_count))
#     logger.info('Semantic accuracy: %.2f%%  (%i/%i)' %
#           (sem_acc, correct_sem, count_sem))
#     logger.info('Syntactic accuracy: %.2f%%  (%i/%i)' %
#           (syn_acc, correct_syn, count_syn))
#     logger.info('Total accuracy: %.2f%%  (%i/%i)' % (tot_acc, correct_tot, count_tot))
#
#     return { "sem" : sem_acc, "syn" : syn_acc, "tot" : tot_acc }
#


sem_ds = ['capital-common-countries', 'capital-world', 'currency', 'city-in-state', 'family']
syn_ds = ['gram1-adjective-to-adverb', 'gram2-opposite', 'gram3-comparative',
          'gram4-superlative', 'gram5-present-participle', 'gram6-nationality-adjective',
          'gram7-past-tense', 'gram8-plural', 'gram9-plural-verbs']

def evaluate_analogies(dictionary, W, g, logger, filter_dictionary, datasets=[]):
    """Evaluate the trained word vectors on a variety of tasks"""

    folder = '/data2/text/analogies_datasets/splitted'

    if not datasets:
        # in each of these folders I expect to find the splits
        datasets = [
            'capital-common-countries', 'capital-world', 'currency',
            'city-in-state', 'family', 'gram1-adjective-to-adverb',
            'gram2-opposite', 'gram3-comparative', 'gram4-superlative',
            'gram5-present-participle', 'gram6-nationality-adjective',
            'gram7-past-tense', 'gram8-plural', 'gram9-plural-verbs',
        ]

    filenames = []

    for dirname in datasets:
        fsplits = glob.glob(os.path.join(folder, os.path.join(dirname, "*.csv")))
        filenames.append(sorted(fsplits))

    filenames = list(itertools.chain.from_iterable(filenames))


    # # to avoid memory overflow, could be increased/decreased
    # # depending on system and vocab size
    # split_size = 100

    full_count = 0  # count all questions, including those with unknown words

    Cq = W
    ICq = np.matmul(g, Cq.T).T
    sqnormq = np.sum(Cq * ICq, axis=1)

    val_dict = {}

    for i in range(len(filenames)):
        label = "-".join(os.path.splitext(filenames[i])[0].split("/")[-2:])

        with open(filenames[i], 'r') as f:
            full_data = [line.rstrip().split(' ') for line in f]
            full_count += len(full_data)
            data = [x for x in full_data if all(word in filter_dictionary for word in x)]

        indices = np.array([[dictionary[word] for word in row] for row in data])
        ind1, ind2, ind3, ind4 = indices.T

        howmany = 1
        predictions = analogies_riemannian(W, g, ind1, ind2, ind3, ICq, sqnormq, howmany)

        # val = (ind4 == predictions)  # correct predictions
        val = [int(i4 in preds) for (i4, preds) in zip(ind4, predictions)]  # correct predictions

        val_dict[label] = val

        logger.info("%s:" % filenames[i])
        logger.info('ACCURACY TOP1: %.2f%% (%d/%d)' %
              (np.mean(val) * 100, np.sum(val), len(val)))

    # categories = ["sem", "syn", "tot"]

    splitkeys = ["0", "1", "2"]
    # aggregate scores for the full datasets
    sem_vals = {}
    syn_vals = {}
    tot_vals = {}

    for split in splitkeys:
        sem_vals[split] = []
        syn_vals[split] = []
        tot_vals[split] = []

    for task, val in val_dict.items():
        d, s = task.split("-split_")

        if d in sem_ds:
            sem_vals[s].append(val)
        elif d in syn_ds:
            syn_vals[s].append(val)
        else:
            raise Exception("{:} is not either semantis or syntactic..".format(d))

        tot_vals[s].append(val)


    all_sem_vals = []
    all_syn_vals = []
    all_tot_vals = []

    acc_dict = {}
    # store aggregated results
    for s in splitkeys:
        label = "sem-split_"+s
        val = np.concatenate(sem_vals[s])
        acc_dict[label] = get_acc_from_val(val)
        all_sem_vals.append(val)

        label = "syn-split_" + s
        val = np.concatenate(syn_vals[s])
        acc_dict[label] = get_acc_from_val(val)
        all_syn_vals.append(val)

        label = "tot-split_" + s
        val = np.concatenate(tot_vals[s])
        acc_dict[label] = get_acc_from_val(val)
        all_tot_vals.append(val)

    val = np.concatenate(all_sem_vals)
    # count_sem = len(val)
    acc_dict["sem"] = get_acc_from_val(val)

    val = np.concatenate(all_syn_vals)
    # count_syn = len(val)
    acc_dict["syn"] = get_acc_from_val(val)

    val = np.concatenate(all_tot_vals)
    # count_tot = len(val)
    acc_dict["tot"] = get_acc_from_val(val)

    # seen_frac = 100 * count_tot / float(full_count)
    # tot_acc = 100 * correct_tot / float(count_tot)
    # sem_acc = 100 * correct_sem / float(count_sem)
    # syn_acc = 100 * correct_syn / float(count_syn)
    #
    # logger.info('Questions seen/total: %.2f%% (%d/%d)' %
    #       (seen_frac, count_tot, full_count))
    # logger.info('Semantic accuracy: %.2f%%  (%i/%i)' %
    #       (sem_acc, correct_sem, count_sem))
    # logger.info('Syntactic accuracy: %.2f%%  (%i/%i)' %
    #       (syn_acc, correct_syn, count_syn))
    # logger.info('Total accuracy: %.2f%%  (%i/%i)' % (tot_acc, correct_tot, count_tot))

    return acc_dict


def get_acc_from_val(val):
    return 100 * sum(val) / len(val)


def analogies_riemannian(embeddings, I, ind1, ind2, ind3, ICq, sqnormq, howmany):
    Ca = embeddings[np.asarray(ind1)]
    Cb = embeddings[np.asarray(ind2)]
    Cc = embeddings[np.asarray(ind3)]

    np.testing.assert_array_equal([len(Ca.shape), len(Cb.shape)], [2,2])
    needed_I_shape = [Ca.shape[1], Cb.shape[1]]
    np.testing.assert_array_equal(needed_I_shape, I.shape)

    Ct = Cb - Ca + Cc

    dists = -2 * np.matmul(Ct, ICq.T) + sqnormq.reshape(1,-1) #+ sqnormt.reshape(-1,1)

    for k in range(len(ind1)):
        dists[k, ind1[k]] = np.Inf
        dists[k, ind2[k]] = np.Inf
        dists[k, ind3[k]] = np.Inf

    all_indexes = np.argmin(dists, axis=1).reshape(-1,1)

    # import pdb; pdb.set_trace()

        # indexes_t = select_indexes(np.argsort(distances),
        #                            to_exclude=[i1, i2, i3],
        #                            howmany=howmany)
        # all_indexes.append(indexes_t)

    return all_indexes

def select_indexes(indexes, to_exclude, howmany):
    return [i for i in indexes if i not in to_exclude][:howmany]


# def analogy_predictions(W, ind1, ind2, ind3, indices, split_size = 500):
#
#     predictions = np.zeros((len(indices),))
#     num_iter = int(np.ceil(len(indices) / float(split_size)))
#     for j in range(num_iter):
#         subset = np.arange(j * split_size, min((j + 1) * split_size, len(ind1)))
#
#         pred_vec = W[ind2[subset], :] - W[ind1[subset], :] + W[ind3[subset], :]
#
#         # cosine similarity if input W has been normalized
#         dist = np.dot(W, pred_vec.T)
#
#         for k in range(len(subset)):
#             dist[ind1[subset[k]], k] = -np.Inf
#             dist[ind2[subset[k]], k] = -np.Inf
#             dist[ind3[subset[k]], k] = -np.Inf
#
#         # predicted word index
#         predictions[subset] = np.argmax(dist, 0).flatten()


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


def center_and_normalize_riemannian(embs, g, center = True, normalize=True):

    if center:
        embs = embs - mean_numexpr(embs, 0)

    if normalize:
        gembs = np.matmul(g, embs.T).T
        norms = np.sqrt(ne.evaluate("embs * gembs").sum(axis=1)).reshape(-1,1)
        embs = ne.evaluate("embs / norms")
        
    return embs

'''
def standardize_riemannian(embs, g):
    normscol = np.sqrt(np.sum(embs**2, axis=0)).reshape(1,-1)
    normscol = normscol * np.diagonal(g).reshape(1,-1)
    embs = ne.evaluate("embs / normscol")

    return embs
'''



def mean_numexpr(embs, index):
    if index==0:
        mean = ne.evaluate("sum(embs, axis=0)") / embs.shape[0]
        return mean.reshape(1, -1)
    elif index==1:
        mean = ne.evaluate("sum(embs, axis=1)") / embs.shape[1]
        return mean.reshape(-1, 1)
    else:
        raise Exception("index can be either 0 or 1")


def merge_dicts(a, b):
    "merges b into a"
    for key in b:
        if isinstance(b[key], dict):
            if (key in a) and isinstance(a[key], dict):
                merge_dicts(a[key], b[key])
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a
