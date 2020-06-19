import argparse
import os
from tqdm import tqdm
import pickle
import numpy as np
import gc
from core.results_collection import load_collection_and_backup, save_collection, check_done_in_collection
from core.measures import evaluate_analogies, merge_dicts, center_and_normalize_eucl,\
    center_and_normalize_riemannian, standardize_riemannian
from mylogging import init_stream_logger, init_logger
from core.load_embeddings import load_embeddings_ldv_hdf, load_dict, get_suffix, \
    compute_p_cIw_from_counts, read_cooccurrences_from_c, load_pretrained_glove, \
    load_glove, get_glove_cc_fnames, load_all_emb_base, get_alpha_ldv_name, get_limit_ldv_name


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
    stream_logger = init_logger(emboutdirname+"/analogies.log")

    vocab, ivocab, vocab_size = load_dict(emb_dir)

    # BASE ANALOGIES
    make_base_analogies(corpus, vecsize, nepoch, vocab, baseoutdir)


    # ALPHA SIMILARITIES
    alphas, I0, Iu, Iud, Ius = load_all_emb_base(emb_dir)

    I0_inv = np.linalg.inv(I0)
    Iu_inv = np.linalg.inv(Iu)
    Iud_inv = np.linalg.inv(Iud)
    Ius_inv = np.linalg.inv(Ius)


    collection_path = emboutdirname+"/alpha-analogies.pkl"
    analogies = load_collection_and_backup(collection_path)

    # 3 points, 6 things, expected 18
    done_alphas, done_limit = check_done_in_collection(analogies, alphas, "u-plog", 18)

    if not done_limit:
        make_limit_analogies(emb_dir, I0, I0_inv, vocab, "u_embeddings", "0", analogies)
        make_limit_analogies(emb_dir, Iu, Iu_inv, vocab, "u_embeddings", "u", analogies)
        make_limit_analogies(emb_dir, Iud, Iud_inv, vocab, "u_embeddings", "ud", analogies)
        save_collection(collection_path, analogies)

    for alpha in tqdm(alphas[done_alphas:], desc="u-alpha-loop"):
        make_alpha_analogies(emb_dir, alpha, I0, I0_inv, vocab, "u_embeddings", "0", analogies)
        make_alpha_analogies(emb_dir, alpha, Iu, Iu_inv, vocab, "u_embeddings", "u", analogies)
        make_alpha_analogies(emb_dir, alpha, Iud, Iud_inv, vocab, "u_embeddings", "ud", analogies)
        save_collection(collection_path, analogies)


    # 4 points, 6 things, expected 24
    done_alphas, done_limit = check_done_in_collection(analogies, alphas, "u+v-plog", 24)

    if not done_limit:
        make_limit_analogies(emb_dir, I0, I0_inv, vocab, "u+v_embeddings", "0", analogies)
        make_limit_analogies(emb_dir, Iu, Iu_inv, vocab, "u+v_embeddings", "u", analogies)
        make_limit_analogies(emb_dir, Ius, Ius_inv, vocab, "u+v_embeddings", "us", analogies)
        make_limit_analogies(emb_dir, Iud, Iud_inv, vocab, "u+v_embeddings", "ud", analogies)
        save_collection(collection_path, analogies)

    for alpha in tqdm(alphas[done_alphas:], desc="u+v-alpha-loop"):
        make_alpha_analogies(emb_dir, alpha, Iu, Iu_inv, vocab, "u+v_embeddings", "u", analogies)
        make_alpha_analogies(emb_dir, alpha, I0, I0_inv, vocab, "u+v_embeddings", "0", analogies)
        make_alpha_analogies(emb_dir, alpha, Iud, Iud_inv, vocab, "u+v_embeddings", "ud", analogies)
        make_alpha_analogies(emb_dir, alpha, Ius, Ius_inv, vocab, "u+v_embeddings", "us", analogies)
        save_collection(collection_path, analogies)


def calc_log_analog(embs, Id, g, vocab, emb_name, analogies = {}):

    method_name = emb_name+"-I"
    analogies_calc_and_org(method_name, embs, Id, vocab=vocab, analogies=analogies)

    method_name = emb_name+"-F"
    analogies_calc_and_org(method_name, embs, g, vocab=vocab, analogies=analogies)

    # method_name = emb_name+"-I+F"
    # analogies_calc_and_org(method_name, embs, (0.5*Id)+g, vocab=vocab, analogies=analogies)


def analogies_calc_and_org(method_name, W, g, vocab, analogies, filter_dictionary=None):

    if filter_dictionary is None:
        filter_dictionary = vocab

    stream_logger.info(method_name)
    stream_logger.info("---------------------------------------------------------------------------")
    acc = evaluate_analogies(vocab, W, g, stream_logger, filter_dictionary)
    stream_logger.info("---------------------------------------------------------------------------")
    stream_logger.info("")

    for task in acc:
        if analogies.get(task, None) is None:
            analogies[task] = {}

        if analogies[task].get(method_name, None) is None:
            analogies[task][method_name] = []

        analogies[task][method_name].append(acc[task])


def make_analogies_fn(emb_dir, ldv_filename, I, I_inv, vocab, emb_name, point_name, analogies, prefix=""):
    stream_logger.info(ldv_filename)
    ldv_path = os.path.join(emb_dir, ldv_filename)
    ldv = load_embeddings_ldv_hdf(ldv_path)
    Id = np.eye(ldv.shape[1])
    plog = np.matmul(I_inv, np.transpose(ldv)).transpose()

    en = emb_name.split("_")[0]
    full_name = prefix + en + "-plog-" + point_name

    calc_log_analog(plog, Id, I, vocab, emb_name=full_name, analogies=analogies)

    # # STANDARDIZE
    # plog_std = standardize_riemannian(plog, I)
    # calc_log_analog(plog_std, Id, I, vocab, emb_name=full_name+"-stF", analogies=analogies)
    #
    # plog_std = standardize_riemannian(plog, Id)
    # calc_log_analog(plog_std, Id, I, vocab, emb_name=full_name+"-stI", analogies=analogies)
    #
    # del plog_std
    # gc.collect()

    # NORMALIZE
    plog_cn = center_and_normalize_riemannian(plog, Id, center=False)
    calc_log_analog(plog_cn, Id, I, vocab, emb_name=full_name+"-nI", analogies=analogies)

    plog_cn = center_and_normalize_riemannian(plog, I, center=False)
    calc_log_analog(plog_cn, Id, I, vocab, emb_name=full_name+"-nF", analogies=analogies)

    # plog_cn = center_and_normalize_riemannian(plog, Id)
    # calc_log_analog(plog_cn, Id, I, vocab, emb_name=full_name+"-cnI", analogies=analogies)
    #
    # plog_cn = center_and_normalize_riemannian(plog, I)
    # calc_log_analog(plog_cn, Id, I, vocab, emb_name=full_name+"-cnF", analogies=analogies)


def make_alpha_analogies(emb_dir, alpha, I, I_inv, vocab, emb_name, point_name, analogies = {}):
    ldv_filename = get_alpha_ldv_name(alpha, emb_name, point_name)
    make_analogies_fn(emb_dir, ldv_filename, I, I_inv, vocab, emb_name, point_name, analogies)


def make_limit_analogies(emb_dir, I, I_inv, vocab, emb_name, point_name, analogies = {}):
    ldv_filename = get_limit_ldv_name(emb_name, point_name)
    make_analogies_fn(emb_dir, ldv_filename, I, I_inv, vocab, emb_name, point_name, analogies, prefix="limit-")


def make_base_analogies(corpus, vecsize, nepoch, dictionary, baseoutdir):

    fnamevc, fnamecc = get_glove_cc_fnames(corpus)

    suffix = get_suffix(vecsize, nepoch)
    gname = corpus + suffix

    outdirname = os.path.join(baseoutdir, gname)

    base_pkl_name = outdirname+"/base-analogies.pkl"

    try:

        with open(base_pkl_name, 'rb') as fstream:
            base_analogs = pickle.load(fstream)

        print("successfully opened base analogies")

    except:

        print("failed to open base analogies, calculating it:")

        base_analogs={}
        calculate_or_load_common_base(fnamecc, dictionary, baseoutdir, base_analogs)

        g_dict, g_vecs, _ = load_glove(corpus, vecsize, nepoch, calc_prob=False)

        Id = np.eye(g_vecs["u"].shape[1])

        analogies_calc_and_org("u", g_vecs["u"], Id,
                               dictionary, analogies=base_analogs)

        # store u vecs, center them and then check u is not changed
        analogies_calc_and_org("u-cn", center_and_normalize_eucl(g_vecs["u"], True, False, 0), Id,
                               dictionary, analogies=base_analogs)


        analogies_calc_and_org("u+v", g_vecs["u"] + g_vecs["v"], Id,
                               dictionary, analogies=base_analogs)
        analogies_calc_and_org("u+v-cn", center_and_normalize_eucl(g_vecs["u"] + g_vecs["v"], True, False, 0), Id,
                                dictionary, analogies=base_analogs)

        print("done base analogies")
        with open(base_pkl_name, 'wb') as fstream:
            pickle.dump(base_analogs, fstream)

    return base_analogs


def calculate_or_load_common_base(fnamecc, dictionary, baseoutdir, corrs):

    _id = os.path.splitext(os.path.basename(fnamecc))[0]

    # NB common base name must depend on the embedding at hand, since the dictionary changes
    common_base = baseoutdir + "/" + _id + "-base_analogies.pkl"

    try:
        print("successfully opened common base")
        with open(common_base, 'rb') as fstream:
            cb_analogs = pickle.load(fstream)

    except:

        print("failed to open common base, calculating it:")
        cb_analogs = {}

        # # stupid method for analogies
        # C = read_cooccurrences_from_c(fnamecc)
        # p_w, p_cIw = compute_p_cIw_from_counts(C)
        # Id = np.eye(p_cIw.shape[1])
        # # def simeval(ind1, ind2):
        # #     return riemannian_cosprod(center_and_normalize_eucl(p_cIw, True, False, 0), Id, ind1, ind2)
        # p_cIw = center_and_normalize_eucl(p_cIw, True, False, 0)
        # analogies_calc_and_org("p_cIw-cn", p_cIw, Id, dictionary, analogies=cb_analogs)
        #
        # del p_cIw, C, p_w
        # gc.collect()
        # # stupid method for analogies

        print("calculating wikigiga5")
        g_dict, g_vecs = load_pretrained_glove("wikigiga5")
        # def simeval(ind1, ind2):
        #     return euclidean_cosprod(center_and_normalize_eucl(g_vecs["u"], True, False, 0), ind1, ind2)
        Id = np.eye(g_vecs["u"].shape[1])
        analogies_calc_and_org("wikigiga5-u+v", g_vecs["u"], Id,
                                g_dict, analogies=cb_analogs, filter_dictionary = dictionary)
        analogies_calc_and_org("wikigiga5-u+v-n", center_and_normalize_eucl(g_vecs["u"], False, False, 0), Id,
                                g_dict, analogies=cb_analogs, filter_dictionary = dictionary)

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
            pickle.dump(cb_analogs, fstream)

    merge_dicts(corrs, cb_analogs)


if __name__ == "__main__":
    main()
