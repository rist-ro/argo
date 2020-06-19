import numpy as np
import scipy
import sys, os
from . import readers
from .spaces import C_numexpr
import numexpr as ne
import tables
import pickle
from gensim.models.keyedvectors import KeyedVectors

import pdb

# GLOVE FILENAMES
info_glove = {
    "wikigiga5" :
        ("/data2/text/word_embeddings/pretrained/glove-pretrained-Wikipedia2014+Gigaword5-6B-v300/glove.6B.300d.txt", 300),
    "commoncrawl42B" :
        ("/data2/text/word_embeddings/pretrained/glove-pretrained-CommonCrawl-42B-v300/glove.42B.300d.txt", 300),
    "commoncrawl840B" :
        ("/data2/text/word_embeddings/pretrained/glove-pretrained-CommonCrawl-840B-v300/glove.840B.300d.txt", 300)
}


default_glove_dir = "/data2/text/word_embeddings"
corpus_name = {

    "geb": "geb",
    "enwiki" : "enwiki201710",
    "swiki": "simplewiki201711"
}

m0 = {
    "geb" : "1000",
    "enwiki" : "1000",
    "swiki": "100"
}

# W2V FILENAMES
info_w2v = {
    "googlenews" :
        ("/data2/text/word_embeddings/pretrained/word2vec-pretrained-GoogleNews100B/GoogleNews-vectors-negative300.bin", 300),
}

info_pretrained = {
    **info_glove,
    **info_w2v
}

def get_glove_cc_fnames(corpus):
    dsdir = '/ssd_data/text/cooccurrences/'
    # simplewiki_sw6_fnamecc = dsdir + 'simplewiki201711/simplewiki201711-sw6-cooccurrence.bin'
    simplewiki_sw10_fnamecc = dsdir + 'simplewiki201711/simplewiki201711-sw10-cooccurrence.bin'
    simplewiki_fnamevc = dsdir + 'simplewiki201711/simplewiki201711-vocab.txt'

    enwiki_sw10_fnamecc = dsdir + 'enwiki201710/enwiki201710-sw10-cooccurrence.bin'
    enwiki_fnamevc = dsdir + 'enwiki201710/enwiki201710-vocab.txt'

    geb_sw10_fnamecc = dsdir + 'geb/geb-sw10-cooccurrence.bin'
    geb_fnamevc = dsdir + 'geb/geb-vocab.txt'

    # select which vocabulary and cooccurrence file to use
    if corpus == "geb":
        fnamevc = geb_fnamevc
        fnamecc = geb_sw10_fnamecc

    elif corpus == "enwiki":
        fnamevc = enwiki_fnamevc
        fnamecc = enwiki_sw10_fnamecc

    elif corpus == "swiki":
        fnamevc = simplewiki_fnamevc
        fnamecc = simplewiki_sw10_fnamecc

    else:
        raise ValueError("corpus not recognized `%s`" % corpus)

    return fnamevc, fnamecc


def normalize_cols_numexpr(mat):
    norms = np.sqrt(ne.evaluate("sum(mat**2, axis=0)")).reshape(1, -1)
    # norms = np.linalg.norm(embs, axis=0)
    # embs = embs / norms
    mat = ne.evaluate("mat / norms")
    return mat


def get_embpath(alpha_tuple, theta, point, emb_dir):
    alpha_id, alpha_kwargs = alpha_tuple
    ldvname = None
    if alpha_id=='float':
        alpha = alpha_kwargs['value']
        ldvname = get_alpha_ldv_name(alpha, theta + "_embeddings", point)
    elif alpha_id=='limit':
        ntop = alpha_kwargs['ntop']
        weighted = alpha_kwargs['weighted']
        fraction = alpha_kwargs['fraction']
        ldvname = get_limit_ldv_name(theta + "_embeddings", point, ntop, weighted, fraction)
    else:
        raise ValueError("Option `{:}`not recognized for alpha_id. Only `float` or `limit` are allowed.".format(alpha_id))

    return os.path.join(emb_dir, ldvname)

# def get_alpha_ldv_name(alpha, emb_name, point_name):
    # if abs(alpha)<0.0001:
    #     ldv_filename = "alpha-{:.2f}-ldv-{}-{:}.hdf".format(abs(alpha), emb_name, point_name)
    # else:
    #     ldv_filename = "alpha{:.2f}-ldv-{}-{:}.hdf".format(alpha, emb_name, point_name)
def get_alpha_ldv_name(alpha, emb_name, point_name, ntop='', weighted=False, frac=False):
    if alpha=="limit":
        return get_limit_ldv_name(emb_name, point_name, ntop=ntop, weighted=weighted, frac=frac)
    ldv_filename = "alpha{:.2f}-ldv-{:}-{:}.hdf".format(alpha, emb_name, point_name)
    return ldv_filename

def get_limit_ldv_name(emb_name, point_name, ntop, weighted, frac):
    ws = ''
    if weighted:
        ws = 'w'

    fs = ''
    if frac:
        fs = 'f'

    ldv_filename = "limit{:}{:}{:}-ldv-{:}-{:}.hdf".format(ntop, ws, fs, emb_name, point_name)
    return ldv_filename


def load_pretrained_glove(gname, dir_name):
    g_reader = readers.get_reader("glove")

    #LOAD GLOVE FROM FILES
    fname, vecsize = dir_name
    consideronlyfirstvec = True

    # u_biases and v_biases are not returned at the moment since we do not know what is the use of them we might have in evaluating word analogies
   
    g_dictionary_size, g_dictionary, g_reversed_dictionary, u_embeddings, v_embeddings = g_reader.read_embeddings(fname, vecsize, consideronlyfirstvec)
    glove_vecs = {"u":u_embeddings, "v":v_embeddings}

    return g_dictionary, glove_vecs

def load_pretrained_w2v(wname):
    # g_reader = readers.get_reader("word2vec")
    fname, vecsize = info_w2v[wname]
    model = KeyedVectors.load_word2vec_format(fname, binary=True)
    vocab = list(model.wv.vocab)
    dictionary = {w:i for i,w in enumerate(vocab)}
    embeddings = np.array([model.get_vector(w) for w in vocab])

    vecs = {"u": embeddings, "v": None}

    return dictionary, vecs


def word_index(word, dictionary):
    try:
        return dictionary[word]
    except KeyError as kerr:
        print("\nKey Error: {0}".format(kerr))
        print("The word requested is not present in the dictionary.\n")
        sys.exit(-1)

def get_glove_fname(base_dir, corpus, vecsize, nepoch):
    nm = corpus_name[corpus]
    gstr = "glove1.2-%s-m0%s-v%d-sw10-lr0.001"%(nm, m0[corpus], vecsize)
    fullname = base_dir + "/" + nm + "/" + gstr + "/" + gstr + "-vectors-n%d.txt" % nepoch
    return fullname

def get_w2v_fname(base_dir, corpus, vecsize, nepoch, sw=10, ns=10):
    nm = corpus_name[corpus]
    gstr = "word2vec-cbow-%s-m0%s-v%d-sw%d-ns%d-lr0.05"%(nm, m0[corpus], vecsize, sw, ns)
    fullname = base_dir + "/" + nm + "/" + gstr + "/" + gstr + "-vectors-n%d.txt" % nepoch
    return fullname

def get_word_embeddings(words, embeddings, dictionary):
    """Return a list of embeddings for the specified words"""
    embeddings = np.asarray(embeddings)
    return np.array([embeddings[word_index(w, dictionary)] for w in words])

def get_suffix(vecsize, nepoch):
    suffix = "-v%d-n%d"%(vecsize, nepoch)
    return suffix

def load_glove(corpus, vecsize, nepoch, calc_prob=False, glove_dir=default_glove_dir):
    g_reader = readers.get_reader("glove")

    #LOAD GLOVE FROM FILES
    fname = get_glove_fname(glove_dir, corpus, vecsize, nepoch)
    consideronlyfirstvec = False

    # u_biases and v_biases are not returned at the moment since we do not know what is the use of them we might have in evaluating word analogies
    g_dictionary_size, g_dictionary, g_reversed_dictionary, u_embeddings, v_embeddings = g_reader.read_embeddings(fname, vecsize, consideronlyfirstvec)
    glove_vecs = {"u" : u_embeddings, "v" : v_embeddings}

    p_tuple = None
    if calc_prob:
        # last word of glove is UNK token
        # U[:-1,:-1], V[:-1,:-1]
        p_tuple = calculate_glove_prob(glove_vecs["u"][:-1,:],
                                       glove_vecs["v"][:-1,:])


    return g_dictionary, glove_vecs, p_tuple


def load_w2v(base_dir, corpus, vecsize, nepoch, sw=10, ns=10):
    reader = readers.get_reader("word2vec")

    fname = get_w2v_fname(base_dir, corpus, vecsize, nepoch, sw=sw, ns=ns)
    consideronlyfirstvec = False
    # u_biases and v_biases are not returned at the moment since we do not know what is the use of them we might have in evaluating word analogies
    dictionary_size, dictionary, reversed_dictionary, u_embeddings, v_embeddings = reader.read_embeddings(fname, vecsize, consideronlyfirstvec)
    vecs = {"u":u_embeddings, "v":v_embeddings}

    return dictionary, vecs

def calculate_glove_prob(U, V, norm_counts_cols = False, pos_measures=False):

    C = C_numexpr(U, V)

    # C = C_numexpr(U, V)

    if norm_counts_cols:
        # C = C / C.sum(axis=0).reshape(1,-1)
        C = normalize_cols_numexpr(C)

    if pos_measures:
        p_cIw = C
        p_w = C.sum(axis=1)
    else:
        N = C.sum()
        sums1 = C.sum(axis=1).reshape(-1, 1)
        p_w = ne.evaluate("sums1 / N")
        p_cIw = ne.evaluate("C / sums1")
        # p_w, p_cIw = compute_p_cIw_from_counts(C)

    return p_w, p_cIw


def calculate_glove_unigram_prob(U, V):

    C = C_numexpr(U, V)
    N = C.sum()
    sums1 = C.sum(axis=1).reshape(-1, 1)
    p_w = ne.evaluate("sums1 / N")

    return p_w


def load_all_emb_base(embdir):
    alphas = np.load(embdir + "/alphas.npy")

    # I = np.load(embdir + "/fisher-" + point_name + ".npy")
    
    I0 = np.load(embdir + "/fisher-0.npy")
    I0_inv = np.linalg.inv(I0)

    Iu = np.load(embdir + "/fisher-u.npy")
    Iu_inv = np.linalg.inv(Iu)

    Iud = np.load(embdir + "/fisher-ud.npy")
    Iud_inv = np.linalg.inv(Iud)

    Ius = np.load(embdir + "/fisher-us.npy")
    Ius_inv = np.linalg.inv(Ius)

    return alphas, I0, Iu, Iud, Ius

def load_emb_base(embdir, point):
    alphas = np.load(embdir + "/alphas.npy")
    I = np.load(embdir + "/fisher-"+point+".npy")
    return alphas, I


def load_embeddings_ldv_hdf(filename, original_norm=False):
    with tables.open_file(filename) as f:
        # print(f.root.embeddings)
        embeddings_ldv = f.root.embeddings_ldv[:, :]

        if original_norm:
            embeddings_lscale = f.root.embeddings_lscale[:, :]
            embeddings_ldv = embeddings_ldv*embeddings_lscale

    return embeddings_ldv


def extract_embedding_from_params(params, vec_size, mode):
    if mode == 0:
        # standard, single embeddings are present
        emb = params
    elif mode == 1:
        # u bu v bu, all parameters are present, I want to use u+v/2
        emb = (params[:vec_size] + params[vec_size+1:-1]) / 2.
    elif mode == 2:
        # u bu v bu, all parameters are present, I want to use u
        emb = params[:vec_size]
    else:
        raise ValueError("mode can be only: 0, 1, or 2. But {:} was found.".format(mode))

    return emb

def save_dict(dictionary, outdirname):
    path = outdirname + "/dictionary.pkl"  # -{:}".format(gname)
    with open(path, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)

def load_dict(vector_dir):
    dict_fname = os.path.join(vector_dir, "dictionary.pkl")
    with open(dict_fname, 'rb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        vocab = pickle.load(f)

    vocab_size = len(vocab)
    ivocab = dict(zip(vocab.values(), vocab.keys()))

    return vocab, ivocab, vocab_size

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
