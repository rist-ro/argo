# export LC_ALL=en_US.UTF-8.

import pickle
from core.measures import evaluate_similarity_on_reverse_split
import numpy as np
import pandas as pd
import argparse
import os

def evaluate_cross_sim_and_org(dictionary, dataset, i_split, dataset_split, method, couples_data, ntop=1, cvcorrs={}):

    p, I_inv, DV, I_norm, I_prod = methods_args[method]

    sorted_data = sorted(couples_data, reverse=True, key=lambda x: x[1])
    top_alphas, top_sims = zip(*sorted_data[:ntop])

    if cvcorrs.get(dataset_split, None) is None:
        cvcorrs[dataset_split] = {}
    if cvcorrs[dataset_split].get(method, None) is None:
        cvcorrs[dataset_split][method] = []

    for alpha in top_alphas:

        simeval = make_simeval(p_embeddings, dictionary, alpha, I_inv, DV,
                                 p, I_norm, I_prod, method="cos")

        corr = evaluate_similarity_on_reverse_split(dictionary, simeval, dataset, i_split)

        cvcorrs[dataset_split][method].append([alpha, corr])

        print("{} alpha {}:".format(dataset_split, alpha))
        print('SPEARMAN CORR: %.2f ' % corr)

def make_simeval(p_embeddings, dictionary, alpha, I_inv, DV,
                                p, I_norm, I_prod, method="cos"):

    def simeval(words1, words2):
        return similarity_logmap_Esubmodel_trick(p_embeddings, dictionary, words1, words2,
                                                 alpha, I_inv, DV, p, I_prod, I_norm=I_norm,
                                                 method=method)

    return simeval


def load_from_dir(simdir):
    alphas = np.load(simdir + "/alphas.npy")

    I0 = np.load(simdir + "/fisher-0.npy")
    # I0_inv = np.linalg.inv(I0)

    Iu = np.load(simdir + "/fisher-u.npy")
    # Iu_inv = np.linalg.inv(Iu)

    with open(simdir+"/base-similarities.pkl", 'rb') as f:
        base_similarities = pickle.load(f)

    with open(simdir+"/alpha-similarities.pkl", 'rb') as f:
        alpha_similarities = pickle.load(f)

    return alphas, I0, Iu, base_similarities, alpha_similarities


def main():

    parser = argparse.ArgumentParser(description='Make cross-validation correlations.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('simdir', type=str, help="directory where to find similarity results")
    parser.add_argument('--ntop', '-t', type=int, help="how many top")

    args = parser.parse_args()

    simdir = args.simdir

    wemb_id = os.path.basename(os.path.normpath(simdir))
    corpus, vstr, nstr = wemb_id.split('-')
    vecsize = int(vstr[1:])
    nepoch = int(nstr[1:])

    ntop = args.ntop

    alphas, I0, Iu, base_similarities, alpha_similarities = load_from_dir(simdir)

    outputname = os.path.join(simdir, "alpha-similarities-cross-val-top{}.json".format(ntop))

    datasets = ["wordsim353",
                "mc", "rg", "scws",
                "wordsim353sim", "wordsim353rel",
                "men", "mturk287", "rw", "simlex999"
                ]


    n_splits = 3

    cvcorrs = {}

    for d in datasets:

        for m in methods_args:

            curves = []

            for n in range(n_splits):
                all_couples = []

                ds = d + "-split_{:}".format(n)

                # load small, mid and large and merge
                for key in data:
                    all_couples += list(zip(alphas[key], data[key][ds][m]))

                all_couples.sort(key=lambda x: x[0])
                all_couples = [(a, v) for a, v in all_couples if np.abs(a)<=70.1]
                all_couples = [(a,v) for a,v in all_couples if not np.isnan(v)]

                #find best top alphas
                # calculate reverse for the selected alpha
                # store results in the form {m: [(a1, s1), (a2, s2),...]}
                evaluate_cross_sim_and_org(v_dictionary, d, n, ds, m, all_couples, ntop, cvcorrs=cvcorrs)


    df = pd.DataFrame(cvcorrs)
    df.to_csv(outputname, sep=' ')


if __name__ == "__main__":
    main()
