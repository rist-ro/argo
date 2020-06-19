import argparse
import numpy as np
import readers, spaces
from spaces import EmbeddingsManager, EuclideanSpace, HyperSphere, calculate_x_embeddings
import os, operator
import multiprocessing
from mylogging import init_logger
from functools import partial
import sys

import os
n_cores = "40"
os.environ["OMP_NUM_THREADS"] = n_cores
# os.environ["OPENBLAS_NUM_THREADS"] = n_cores
# os.environ["MKL_NUM_THREADS"] = n_cores
# os.environ["VECLIB_MAXIMUM_THREADS"] = n_cores
os.environ["NUMEXPR_NUM_THREADS"] = n_cores

import numexpr as ne
ne.set_num_threads(40)

parser = argparse.ArgumentParser(description='Evaluate analogies on a word embedding.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('inputfile', help='The file where to find the alpha embeddings.')
parser.add_argument('--vecsize', '-v', required=True, type=int, help='Vector size of the alpha embeddings.')
parser.add_argument('--outputfolder', '-o', required=True, help='folder where to save output stats')
parser.add_argument('--vocabulary', help='Optionally, specify the inputfile from which to read the vocabulary to use. \
        Default uses the vocabulary of the inputfile. (same naming convention as for the inputfile is expected)', default=None)
# parser.add_argument('--wordcounts', help='Optionally, specify the file from which to read the wordcounts used to calculate the unigram\
#                     distribution (it should contain counts for the dictionary)', default=None)
# parser.add_argument('--amonghowmany', '-ahm', type=int, help='How many words of the dictionary to consider as candidates, i.e. among how many words to consider the possible "nearest words" to the analogy. -1 for all of them.', default=-1)
parser.add_argument('--tolerance', action='store_true', help='If this flag is set, the words in the query are removed from the result.')
# parser.add_argument('--numthreads', '-nt', type=int, help='Number of threads to use.', default=40)

# subparsers = parser.add_subparsers(dest='subcommand')
# subparsers.required = True
#
# linear_parser = subparsers.add_parser('linear', help='Evaluate accuracies based on the assumption of a linear space, consider each line in the inputfile: word u_vec')
# sphere_parser = subparsers.add_parser('sphere', help='Evaluate accuracies based on the geometry of the sphere, consider each line in the inputfile: word u_vec u_bias v_vec v_bias')

def word_index(word, dictionary):
    try:
        return dictionary[word]
    except KeyError as kerr:
        print("\nKey Error: {0}".format(kerr))
        print("The word requested is not present in the dictionary.\n")
        sys.exit(-1)

def get_word_embeddings(words, embeddings, dictionary):
    """Return a list of embeddings for the specified words"""
    embeddings = np.asarray(embeddings)
    return np.array([embeddings[word_index(w, dictionary)] for w in words])

def get_word_indexes(words, dictionary):
    """Return a list of indexes for the specified words"""
    return np.array([word_index(w, dictionary) for w in words])

def analogies_euclidean(embeddings, dictionary, words1, words2, words3, howmany, tolerance):
    emb1 = get_word_embeddings(words1, embeddings, dictionary)
    ind1 = get_word_indexes(words1, dictionary)

    emb2 = get_word_embeddings(words2, embeddings, dictionary)
    ind2 = get_word_indexes(words2, dictionary)

    emb3 = get_word_embeddings(words3, embeddings, dictionary)
    ind3 = get_word_indexes(words3, dictionary)

    emb_targets = emb2 - emb1 + emb3

    all_indexes = []

    for embt, i1, i2, i3 in zip(emb_targets, ind1, ind2, ind3):
        diffs = embt.reshape(1,-1) - embeddings
        # import pdb;pdb.set_trace()
        distances = ne.evaluate('diffs**2').sum(axis=1)
        # import pdb;pdb.set_trace()
        # distances = np.sqrt(distances)
        # indexes_t = np.argsort(distances)[:howmany]

        indexes_t = select_indexes(np.argsort(distances),
                                   to_exclude=[i1, i2, i3],
                                   howmany=howmany)
        all_indexes.append(indexes_t)

    # emb_targets = np.expand_dims(emb_targets, axis=1)
    # embeddings = np.expand_dims(embeddings, axis=0)
    #
    # # distances = np.linalg.norm(emb_targets-embeddings, axis=2)
    # # diff = emb_targets-embeddings
    # # distances = ne.evaluate('sum(diff*diff, axis=2)')
    # distances = ne.evaluate('(emb_targets-embeddings)**2').sum(axis=2)
    # # import pdb;pdb.set_trace()
    # # distances = np.sum((emb_targets-embeddings)**2, axis=2)
    # # distances = np.sqrt(distances)
    # all_indexes = np.argsort(distances, axis=1)

    # if tolerance:
    #     all_indexes = [select_indexes(query_pred_indexes, [i1, i2, i3], howmany)
    #                        for query_pred_indexes, i1, i2, i3 in zip(all_indexes, ind1, ind2, ind3)]
    # else:
    #     all_indexes = all_indexes[:, :howmany]

    return all_indexes

def select_indexes(indexes, to_exclude, howmany):
    return [i for i in indexes if i not in to_exclude][:howmany]

def evaluate_analogies(aneval, logger, tolerance):
    """Evaluate the trained word vectors on a variety of analogies"""

    folder='/data/captamerica_hd2/text/analogies_datasets'
    filenames = [
            'capital-common-countries.txt', 'capital-world.txt', 'currency.txt',
            'city-in-state.txt', 'family.txt', 'gram1-adjective-to-adverb.txt',
            'gram2-opposite.txt', 'gram3-comparative.txt', 'gram4-superlative.txt',
            'gram5-present-participle.txt', 'gram6-nationality-adjective.txt',
            'gram7-past-tense.txt', 'gram8-plural.txt', 'gram9-plural-verbs.txt',
            ]
    
    topnumbers = [1, 5, 10]
    howmany = max(topnumbers)
    # # to avoid memory overflow, could be increased/decreased
    # # depending on system and vocab size
    # split_size = 100
    
    count_sem = 0 # count all semantic questions
    count_syn = 0 # count all syntactic questions
    count_tot = 0 # count all questions
    full_count = 0 # count all questions, including those with unknown words

    correct_tot = {} # count correct semantic questions
    correct_sem = {} # count correct syntactic questions
    correct_syn = {} # count correct questions

    for tnum in topnumbers:
        correct_tot[tnum] = 0
        correct_sem[tnum] = 0
        correct_syn[tnum] = 0

    for i in range(len(filenames)):
        with open('%s/%s' % (folder, filenames[i]), 'r') as f:
            full_data = [line.rstrip().split(' ') for line in f]
            full_count += len(full_data)
            data = [x for x in full_data if all(word in dictionary for word in x)]

        # import pdb;pdb.set_trace()
        words1, words2, words3, words4 = zip(*data)

        # indices = np.array([[dictionary[word] for word in row] for row in data])
        # ind1, ind2, ind3, ind4 = indices.T
        #
        # analogy_query_d_from_indexes = partial(embeddings_manager.analogy_query_d_from_indexes,
        #                                 howmany=howmany, amonghowmany=amonghowmany, tolerance=tolerance)
        #
        # return `howmany` couples [(index, measure)] for each query
        # index_and_measures = pool.starmap(analogy_query_d_from_indexes, zip(ind1, ind2, ind3))

        # for each query extract the list of index predictions
        # predictions = [list(zip(*iam))[0] for iam in index_and_measures]
        # import pdb; pdb.set_trace()
        predictions = aneval(words1, words2, words3, howmany=howmany, tolerance=tolerance)
        ind4 = [dictionary[w] for w in words4]

        logger.info("%s:" % filenames[i])
        
        count_tot = count_tot + len(words1)

        if i < 5:
            count_sem = count_sem + len(words1)
        else:
            count_syn = count_syn + len(words1)
        
        val = {}
        # import pdb;pdb.set_trace()
        for tnum in topnumbers:
            val[tnum] = [int(i4 in preds[:tnum]) for (i4,preds) in zip(ind4, predictions)] # correct predictions
            
            correct_tot[tnum] = correct_tot[tnum] + np.sum(val[tnum])
            
            if i < 5:
                correct_sem[tnum] = correct_sem[tnum] + np.sum(val[tnum])
            else:
                correct_syn[tnum] = correct_syn[tnum] + np.sum(val[tnum])

            logger.info('ACCURACY TOP%d: %.2f%% (%d/%d)' %
                (tnum, np.mean(val[tnum])*100, np.sum(val[tnum]), len(val[tnum])))

    logger.info('Questions seen/total: %.2f%% (%d/%d)' %
        (100 * count_tot / float(full_count), count_tot, full_count))
    
    for tnum in topnumbers:
        logger.info('\nTOP%d STATS'%tnum)
        logger.info('-------------------------')
        logger.info('Semantic accuracy: %.2f%%  (%i/%i)' %
            (100 * correct_sem[tnum] / float(count_sem), correct_sem[tnum], count_sem))
        logger.info('Syntactic accuracy: %.2f%%  (%i/%i)' %
            (100 * correct_syn[tnum] / float(count_syn), correct_syn[tnum], count_syn))
        logger.info('Total accuracy: %.2f%%  (%i/%i)' %
            (100 * correct_tot[tnum] / float(count_tot), correct_tot[tnum], count_tot))


args = parser.parse_args()

inputlabel = readers.rmtxt(os.path.basename(args.inputfile))
outputfolder=args.outputfolder
if outputfolder.endswith('/'):
  outputfolder=outputfolder[:-1]

outputdatafolder = outputfolder+'/'+inputlabel
os.makedirs(outputdatafolder, exist_ok=True)

tolerance=args.tolerance
tolstr=''
if tolerance:
    tolstr='-tol'

analogiesoutputname=outputdatafolder+'/analogies'+tolstr+'.txt'
analogieslogger=init_logger(analogiesoutputname)

reader=readers.get_reader("glove")
words_set=None
if args.vocabulary:
    (dictionary_size, dictionary, reversed_dictionary) = \
        reader.read_dictionary(args.vocabulary)
    words_set=set(dictionary.keys())

# consideronlyfirstvec=None
# if args.subcommand=='linear':
consideronlyfirstvec=True
# elif args.subcommand=='sphere':
#     consideronlyfirstvec=False

# u_biases and v_biases are not returned at the moment since we do not know what is the use of them we might have in evaluating word analogies
dictionary_size, dictionary, reversed_dictionary, u_embeddings, v_embeddings = \
    reader.read_embeddings(args.inputfile, args.vecsize, consideronlyfirstvec, words_set=words_set)

def aneval(words1, words2, words3, howmany, tolerance):
    return analogies_euclidean(u_embeddings, dictionary, words1, words2, words3, howmany=howmany, tolerance=tolerance)

evaluate_analogies(aneval, analogieslogger, tolerance)
