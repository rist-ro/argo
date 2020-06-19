import argparse
import numpy as np
# import scipy
from core.results_collection import load_collection_and_backup, save_collection, check_done_in_collection
from core.measures import evaluate_similarities, center_and_normalize_eucl, merge_dicts,\
    center_and_normalize_riemannian, standardize_riemannian
from core.load_embeddings import load_embeddings_ldv_hdf, load_dict, get_suffix, \
    compute_p_cIw_from_counts, read_cooccurrences_from_c, load_pretrained_glove, \
    load_glove, get_glove_cc_fnames, get_alpha_ldv_name, get_limit_ldv_name, load_all_emb_base

import os
import pickle
from mylogging import init_stream_logger, init_logger
# stream_logger = init_stream_logger()
import gc
from tqdm import tqdm

# init_logger(outputfile)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('embdir', type=str, help="directory where to find structure of embeddings")
    parser.add_argument('--outputdir', '-o', help='outputdir')

    args = parser.parse_args()

    emb_dir = args.embdir
    baseoutdir = args.outputdir

    emb_id = os.path.basename(os.path.normpath(emb_dir))
    corpus, vstr, nstr = emb_id.split('-')
    vecsize = int(vstr[1:])
    nepoch = int(nstr[1:])

    emboutdirname = os.path.join(baseoutdir, emb_id)
    os.makedirs(emboutdirname, exist_ok=True)

    global stream_logger
    stream_logger = init_logger(emboutdirname+"/similarities.log")

    vocab, ivocab, vocab_size = load_dict(emb_dir)

    # BASE SIMILARITIES
    make_base_sims(corpus, vecsize, nepoch, vocab, baseoutdir)

    # ALPHA SIMILARITIES
    alphas, I0, Iu, Iud, Ius = load_all_emb_base(emb_dir)

    I0_inv = np.linalg.inv(I0)
    Iu_inv = np.linalg.inv(Iu)
    Iud_inv = np.linalg.inv(Iud)
    Ius_inv = np.linalg.inv(Ius)


    collection_path = emboutdirname+"/alpha-similarities.pkl"
    correlations = load_collection_and_backup(collection_path)

    # 3 points, 6 things, expected 18
    done_alphas, done_limit = check_done_in_collection(correlations, alphas, "u-plog", 18)

    if not done_limit:
        make_limit_sims(emb_dir, I0, I0_inv, vocab, "u_embeddings", "0", correlations)
        make_limit_sims(emb_dir, Iu, Iu_inv, vocab, "u_embeddings", "u", correlations)
        make_limit_sims(emb_dir, Iud, Iud_inv, vocab, "u_embeddings", "ud", correlations)
        save_collection(collection_path, correlations)

    for alpha in tqdm(alphas[done_alphas:], desc="u-alpha-loop"):
        make_alpha_sims(emb_dir, alpha, I0, I0_inv, vocab, "u_embeddings", "0", correlations)
        make_alpha_sims(emb_dir, alpha, Iu, Iu_inv, vocab, "u_embeddings", "u", correlations)
        make_alpha_sims(emb_dir, alpha, Iud, Iud_inv, vocab, "u_embeddings", "ud", correlations)
        save_collection(collection_path, correlations)


    # 4 points, 6 things, expected 24
    done_alphas, done_limit = check_done_in_collection(correlations, alphas, "u+v-plog", 24)

    if not done_limit:
        make_limit_sims(emb_dir, I0, I0_inv, vocab, "u+v_embeddings", "0", correlations)
        make_limit_sims(emb_dir, Iu, Iu_inv, vocab, "u+v_embeddings", "u", correlations)
        make_limit_sims(emb_dir, Ius, Ius_inv, vocab, "u+v_embeddings", "us", correlations)
        make_limit_sims(emb_dir, Iud, Iud_inv, vocab, "u+v_embeddings", "ud", correlations)
        save_collection(collection_path, correlations)

    for alpha in tqdm(alphas[done_alphas:], desc="u+v-alpha-loop"):
        make_alpha_sims(emb_dir, alpha, Iu, Iu_inv, vocab, "u+v_embeddings", "u", correlations)
        make_alpha_sims(emb_dir, alpha, I0, I0_inv, vocab, "u+v_embeddings", "0", correlations)
        make_alpha_sims(emb_dir, alpha, Iud, Iud_inv, vocab, "u+v_embeddings", "ud", correlations)
        make_alpha_sims(emb_dir, alpha, Ius, Ius_inv, vocab, "u+v_embeddings", "us", correlations)
        save_collection(collection_path, correlations)



def similarity_calc_and_org(method_name, W, g, vocab, correlations, filter_dictionary = None):

    if filter_dictionary is None:
        filter_dictionary = vocab

    stream_logger.info(method_name)
    stream_logger.info("---------------------------------------------------------------------------")
    sc, corr = evaluate_similarities(vocab, W, g, stream_logger, filter_dictionary, datasets = [], plot=False)
    stream_logger.info("---------------------------------------------------------------------------")
    stream_logger.info("")

    for task in corr:
        if correlations.get(task, None) is None:
            correlations[task] = {}

        if correlations[task].get(method_name, None) is None:
            correlations[task][method_name] = []

        correlations[task][method_name].append(corr[task])

# def similarity_calc_and_org(method_name, simeval, vocab, correlations):
#
#     stream_logger.info(method_name)
#     stream_logger.info("---------------------------------------------------------------------------")
#     sc, corr = evaluate_similarities(vocab, simeval, stream_logger, datasets = [], plot=False)
#     stream_logger.info("---------------------------------------------------------------------------")
#     stream_logger.info("")
#
#     for task in corr:
#         if correlations.get(task, None) is None:
#             correlations[task] = {}
#
#         if correlations[task].get(method_name, None) is None:
#             correlations[task][method_name] = []
#
#         correlations[task][method_name].append(corr[task])
#

def calc_log_sims(embs, Id, g, vocab, emb_name, correlations = {}):

    method_name = emb_name+"-I"
    similarity_calc_and_org(method_name, embs, Id, vocab=vocab, correlations=correlations)

    method_name = emb_name+"-F"
    similarity_calc_and_org(method_name, embs, g, vocab=vocab, correlations=correlations)

    # method_name = emb_name+"-I+F"
    # similarity_calc_and_org(method_name, embs, (0.5*Id)+g, vocab=vocab, correlations=correlations)


def make_sims_fn(ldv_filename, emb_dir, I, I_inv, vocab, emb_name, point_name, correlations, prefix=""):
    stream_logger.info(ldv_filename)

    ldv_path = os.path.join(emb_dir, ldv_filename)
    ldv = load_embeddings_ldv_hdf(ldv_path)
    Id = np.eye(ldv.shape[1])
    plog = np.matmul(I_inv, np.transpose(ldv)).transpose()

    en = emb_name.split("_")[0]
    full_name = prefix + en + "-plog-" + point_name

    # ldv commented
    # calc_log_sims(ldv, Id, I_inv, vocab, emb_name="ldv-"+point_name+"-st0", correlations=correlations)
    # ldv commented

    calc_log_sims(plog, Id, I, vocab, emb_name=full_name, correlations=correlations)

    # STANDARDIZE FROM HERE
    # ldv commented
    # d0 = (np.sum(ldv ** 2, axis=0) ** (0.5))
    # ldv_std = ldv / d0.reshape(1, -1)
    # calc_log_sims(ldv_std, Id, I_inv, vocab, emb_name="ldv-"+point_name+"-st1", correlations=correlations)
    # ldv commented

    # # STANDARDIZE
    # plog_std = standardize_riemannian(plog, I)
    # calc_log_sims(plog_std, Id, I, vocab, emb_name=full_name+"-stF", correlations=correlations)
    #
    # plog_std = standardize_riemannian(plog, Id)
    # calc_log_sims(plog_std, Id, I, vocab, emb_name=full_name+"-stI", correlations=correlations)
    #
    # del plog_std
    # gc.collect()

    # NORMALIZE
    plog_cn = center_and_normalize_riemannian(plog, Id)
    calc_log_sims(plog_cn, Id, I, vocab, emb_name=full_name+"-cnI", correlations=correlations)

    plog_cn = center_and_normalize_riemannian(plog, I)
    calc_log_sims(plog_cn, Id, I, vocab, emb_name=full_name+"-cnF", correlations=correlations)

    # only normalize for similarities does not make sense, since it will always use cos product after
    # plog_cn = center_and_normalize_riemannian(plog, Id, center=False)
    # calc_log_sims(plog_cn, Id, I, vocab, emb_name=prefix + "plog-" + point_name + "-nI", correlations=correlations)
    #
    # plog_cn = center_and_normalize_riemannian(plog, I, center=False)
    # calc_log_sims(plog_cn, Id, I, vocab, emb_name=prefix + "plog-" + point_name + "-nF", correlations=correlations)


def make_alpha_sims(emb_dir, alpha, I, I_inv, vocab, emb_name, point_name, correlations = {}):
    # point_name is either "0" or "u"
    ldv_filename = get_alpha_ldv_name(alpha, emb_name, point_name)
    make_sims_fn(ldv_filename, emb_dir, I, I_inv, vocab, emb_name, point_name, correlations)


def make_limit_sims(emb_dir, I, I_inv, vocab, emb_name, point_name, correlations = {}):
    # point_name is either "0" or "u"
    ldv_filename = get_limit_ldv_name(emb_name, point_name)
    make_sims_fn(ldv_filename, emb_dir, I, I_inv, vocab, emb_name, point_name, correlations, prefix="limit-")


def make_base_sims(corpus, vecsize, nepoch, dictionary, baseoutdir):

    fnamevc, fnamecc = get_glove_cc_fnames(corpus)

    suffix = get_suffix(vecsize, nepoch)
    gname = corpus + suffix

    outdirname = os.path.join(baseoutdir, gname)

    base_pkl_name = outdirname+"/base-similarities.pkl"

    try:
        with open(base_pkl_name, 'rb') as fstream:
            base_sims = pickle.load(fstream)

    except:

        base_sims={}
        calculate_or_load_common_base(fnamecc, dictionary, baseoutdir, base_sims)

        g_dict, g_vecs, _ = load_glove(corpus, vecsize, nepoch, calc_prob=False)

        Id = np.eye(g_vecs["u"].shape[1])
        # def simeval(ind1, ind2):
        #     return riemannian_cosprod(center_and_normalize_eucl(g_vecs["u"], True, False, 0), Id, ind1, ind2)
        similarity_calc_and_org("u-cn", center_and_normalize_eucl(g_vecs["u"], True, False, 0), Id, dictionary,
                                correlations=base_sims)

        # def simeval(ind1, ind2):
        #     return euclidean_cosprod(center_and_normalize_eucl(g_vecs["u"] + g_vecs["v"], True, False, 0), ind1, ind2)

        similarity_calc_and_org("u+v-cn", center_and_normalize_eucl(g_vecs["u"] + g_vecs["v"], True, False, 0), Id,
                                dictionary, correlations=base_sims)


        with open(base_pkl_name, 'wb') as fstream:
            pickle.dump(base_sims, fstream)

    return base_sims


def calculate_or_load_common_base(fnamecc, dictionary, baseoutdir, corrs):

    _id = os.path.splitext(os.path.basename(fnamecc))[0]

    # NB common base name must depend on the embedding at hand, since the dictionary changes
    common_base = baseoutdir + "/" + _id + "-base_similarities.pkl"

    try:
        with open(common_base, 'rb') as fstream:
            cb_corrs = pickle.load(fstream)

        print("successfully opened common base")

    except:

        cb_corrs = {}

        print("failed to open common base, calculating it:")

        print("calculating pcIw")
        C = read_cooccurrences_from_c(fnamecc)
        p_w, p_cIw = compute_p_cIw_from_counts(C)
        Id = np.eye(p_cIw.shape[1])
        # def simeval(ind1, ind2):
        #     return riemannian_cosprod(center_and_normalize_eucl(p_cIw, True, False, 0), Id, ind1, ind2)
        p_cIw = center_and_normalize_eucl(p_cIw, True, False, 0)
        similarity_calc_and_org("p_cIw-cn", p_cIw, Id, dictionary, correlations=cb_corrs)

        del p_cIw, C, p_w
        gc.collect()

        print("calculating wikigiga5")
        g_dict, g_vecs = load_pretrained_glove("wikigiga5")
        # def simeval(ind1, ind2):
        #     return euclidean_cosprod(center_and_normalize_eucl(g_vecs["u"], True, False, 0), ind1, ind2)
        Id = np.eye(g_vecs["u"].shape[1])
        similarity_calc_and_org("wikigiga5-u+v", center_and_normalize_eucl(g_vecs["u"], False, False, 0), Id,
                                g_dict, correlations=cb_corrs, filter_dictionary=dictionary)

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

        print("done common base")

        with open(common_base, 'wb') as fstream:
            pickle.dump(cb_corrs, fstream)


    merge_dicts(corrs, cb_corrs)


if __name__ == "__main__":
    main()
