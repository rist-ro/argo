import argparse
import os
import gc
from functools import partial
from core.load_embeddings import get_suffix, get_word_embeddings, load_glove, calculate_glove_prob, calculate_glove_unigram_prob, save_dict, get_alpha_ldv_name
from core.compute_embeddings import NUMTOL, fisher_matrix_and_whatnot, project_on_basis_from_ps, scalar_prod_logp0pw_beta_basis_npf
import numpy as np
import tables
from core import readers
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from itertools import product

ntops = [1, 3, 5]
weigthed_limits = [True, False]
frac_limits = [True, False]

def calculate_all_embeddings(U, V, pud, alphas, dictionary, outdirname):

    g_p_w, g_p_cIw = calculate_glove_prob(U, V)
    damping_couples = []

    gname = os.path.basename(outdirname)

    # uniform
    p0 = np.ones(V.shape[0]) / V.shape[0]
    I0, DV0, _ = well_cond_fisher_and_whatnot(V, p0, "F-0", damping_couples=damping_couples)

    # unigram from data
    Iud, DVud, _ = well_cond_fisher_and_whatnot(V, pud, "F-ud", damping_couples=damping_couples)

    # unigram from glove model theta=u
    pu = g_p_w
    Iu, DVu, _ = well_cond_fisher_and_whatnot(V, pu, "F-u", damping_couples=damping_couples)

    # save dictionary
    save_dict(dictionary, outdirname)
    # save Fisher metric 0
    np.save(outdirname + "/fisher-0", I0)
    # save Fisher metric u
    np.save(outdirname + "/fisher-u", Iu)
    # save Fisher metric u from data
    np.save(outdirname + "/fisher-ud", Iud)
    # save alphas
    np.save(outdirname + "/alphas", alphas)

    # MODEL FROM THETA = U
    p_embeddings = g_p_cIw

    # limits ldv
    for ntop, wbool, fbool in tqdm(list(product(ntops, weigthed_limits, frac_limits)), desc="{:} limit-u".format(gname)):
        # output = outdirname + "/limit{:d}-ldv-u_embeddings-0".format(ntop)
        limit_embeddings_ldv(dictionary, p_embeddings, p0, DV0, outdirname, "u_embeddings", "0", ntop=ntop, weighted=wbool, frac=fbool)
        # output = outdirname + "/limit-ldv-u_embeddings-u"
        limit_embeddings_ldv(dictionary, p_embeddings, pu, DVu, outdirname, "u_embeddings", "u", ntop=ntop, weighted=wbool, frac=fbool)
        # output = outdirname + "/limit{:d}-ldv-u_embeddings-ud"
        limit_embeddings_ldv(dictionary, p_embeddings, pud, DVud, outdirname, "u_embeddings", "ud", ntop=ntop, weighted=wbool, frac=fbool)

    # save ldv 0
    for alpha in tqdm(alphas, desc="{:} alpha-u-0".format(gname)):
        # output = outdirname + "/alpha{:.2f}-ldv-u_embeddings-0".format(alpha)
        alpha_embeddings_ldv(alpha, dictionary, p_embeddings, p0, DV0, outdirname, "u_embeddings", "0")

    # save ldv u
    for alpha in tqdm(alphas, desc="{:} alpha-u-u".format(gname)):
        # output = outdirname + "/alpha{:.2f}-ldv-u_embeddings-u".format(alpha)
        alpha_embeddings_ldv(alpha, dictionary, p_embeddings, pu, DVu, outdirname, "u_embeddings", "u")

    # save ldv ud
    for alpha in tqdm(alphas, desc="{:} alpha-u-ud".format(gname)):
        # output = outdirname + "/alpha{:.2f}-ldv-u_embeddings-ud".format(alpha)
        alpha_embeddings_ldv(alpha, dictionary, p_embeddings, pud, DVud, outdirname, "u_embeddings", "ud")

    #free some memory
    del p_embeddings, g_p_cIw
    gc.collect()

    # MODEL FROM THETA = U+V
    g_p_w, g_p_cIw = calculate_glove_prob(U+V, V)

    # unigram from glove model theta=u
    pus = g_p_w
    Ius, DVus, _ = well_cond_fisher_and_whatnot(V, pus, "F-us", damping_couples=damping_couples)

    p_embeddings = g_p_cIw

    # plot eigenvalues of fisher metric in these parametrizations
    plot_fisher_eigs([I0, Iu, Iud, Ius], ["F-0", "F-u", "F-ud", "F-us"], outdirname)
    print_damping(damping_couples, outdirname)

    # save Fisher metric u from sum model
    np.save(outdirname + "/fisher-us", Ius)

    # limits ldv
    for ntop, wbool, fbool in tqdm(list(product(ntops, weigthed_limits, frac_limits)), desc="{:} limit-u+v".format(gname)):
        # output = outdirname + "/limit-ldv-u+v_embeddings-0"
        limit_embeddings_ldv(dictionary, p_embeddings, p0, DV0, outdirname, "u+v_embeddings", "0", ntop=ntop, weighted=wbool, frac=fbool)
        # output = outdirname + "/limit-ldv-u+v_embeddings-u"
        limit_embeddings_ldv(dictionary, p_embeddings, pu, DVu, outdirname, "u+v_embeddings", "u", ntop=ntop, weighted=wbool, frac=fbool)
        # output = outdirname + "/limit-ldv-u+v_embeddings-ud"
        limit_embeddings_ldv(dictionary, p_embeddings, pud, DVud, outdirname, "u+v_embeddings", "ud", ntop=ntop, weighted=wbool, frac=fbool)
       # output = outdirname + "/limit-ldv-u+v_embeddings-us"
        limit_embeddings_ldv(dictionary, p_embeddings, pus, DVus, outdirname, "u+v_embeddings", "us", ntop=ntop, weighted=wbool, frac=fbool)

    # save ldv 0
    for alpha in tqdm(alphas, desc="{:} alpha-u+v-0".format(gname)):
        # output = outdirname + "/alpha{:.2f}-ldv-u+v_embeddings-0".format(alpha)
        alpha_embeddings_ldv(alpha, dictionary, p_embeddings, p0, DV0, outdirname, "u+v_embeddings", "0")

    # save ldv u
    for alpha in tqdm(alphas, desc="{:} alpha-u+v-u".format(gname)):
        # output = outdirname + "/alpha{:.2f}-ldv-u+v_embeddings-u".format(alpha)
        alpha_embeddings_ldv(alpha, dictionary, p_embeddings, pu, DVu, outdirname, "u+v_embeddings", "u")

    # save ldv ud
    for alpha in tqdm(alphas, desc="{:} alpha-u+v-ud".format(gname)):
        # output = outdirname + "/alpha{:.2f}-ldv-u+v_embeddings-ud".format(alpha)
        alpha_embeddings_ldv(alpha, dictionary, p_embeddings, pud, DVud, outdirname, "u+v_embeddings", "ud")

    # temporarily commented, not clear usefulness, usually us is not very good...
    # # save ldv us
    # for alpha in tqdm(alphas, desc="alpha-us"):
    #     output = outdirname + "/alpha{:.2f}-ldv-u+v_embeddings-us".format(alpha)
    #     alpha_embeddings_ldv(alpha, dictionary, p_embeddings, pus, DVus, output)


def well_cond_fisher_and_whatnot(V, p, nameI, damping_couples=[]):
    max_cond = 1e3
    p = p.reshape(-1, 1)
    I, DV = fisher_matrix_and_whatnot(V, p)
    p = p.reshape(1, -1)

    damping = 0
    condition = np.linalg.cond(I)

    good = False
    if condition<=max_cond:
        good = True
    else:
        damping = 1e-6

    while not good:
        I, DV = fisher_matrix_and_whatnot(V, p, damping=damping)
        condition = np.linalg.cond(I)
        good = (condition<=max_cond) and (condition>=0.9*max_cond)
        if not good:
            if condition>max_cond:
                damping *= 10
            elif condition<0.9*max_cond:
                damping /= 2
            else:
                raise Exception("Not sure what is happening here. "
                                "Condition is {:} and max_cond is {:}".format(condition, max_cond))

    print("{:} : had to use damping factor of {:}, for condition number {:}".format(nameI, damping, condition))
    damping_couples.append((nameI, damping))

    return I, DV, damping_couples


def plot_fisher_eigs(I_list, labels, outdirname):
    # isotropicity of v vectors
    for I,label in zip(I_list, labels):
        eigs, V = np.linalg.eigh(I)
        eigs = np.array(sorted(eigs, reverse=True))
        plt.plot(np.log(eigs), label=label)
        #plt.plot(eigs/eigs[0], label=label)

    plt.legend()
    plt.savefig(outdirname+"/fisher_eigenvalues.png")


def print_damping(damping_couples, outdirname):
    with open(outdirname+"/damping.txt", 'w') as out_file:
        for couple in damping_couples:
            out_file.write(" ".join(map(str, couple))+"\n")


def alpha_embeddings_txt(alpha, dictionary, p_embeddings, pu, DV, I_inv, output, chunk_size=1000):
    # print("calculating vectors for alpha={:.1f}".format(alpha))
    all_words = list(dictionary.keys())

    fstream = open(output, "w")

    for i in range(0, len(all_words), chunk_size):
        words = all_words[i: i + chunk_size]

        # proj logmap
        p1 = get_word_embeddings(words, p_embeddings, dictionary)
        # u1 = get_word_embeddings(words1, u_embeddings, dictionary)
        # v1 = get_word_embeddings(words1, v_embeddings, dictionary)

        C_proj1 = project_on_basis_from_ps(p1, DV, I_inv, pu, alpha)

        write_to_file(fstream, words, C_proj1)
    fstream.close()

def alpha_embeddings_ldv(alpha, dictionary, p_embeddings, p0, DV, outdirname, emb_name, point_name, chunk_size=5000):
    # print("calculating vectors for alpha={:.1f}".format(alpha))

    full_path = os.path.join(outdirname, get_alpha_ldv_name(alpha, emb_name, point_name))

    #check if file already exists and is not empty (nothing to do)
    if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
        return

    all_words = list(dictionary.keys())

    with tables.open_file(full_path, 'w') as f:
        atom = tables.Atom.from_dtype(p_embeddings.dtype)

        vec_size = DV.shape[1]
        # 0 means the array can be expanded on that axis
        array_ldv = f.create_earray(f.root, 'embeddings_ldv', atom, (0, vec_size))
        array_lscale = f.create_earray(f.root, 'embeddings_lscale', atom, (0, 1))

        for i in range(0, len(all_words), chunk_size):
            words = all_words[i: i + chunk_size]

            # proj logmap
            p1 = get_word_embeddings(words, p_embeddings, dictionary)
            # u1 = get_word_embeddings(words1, u_embeddings, dictionary)
            # v1 = get_word_embeddings(words1, v_embeddings, dictionary)

            # scalprods_ldv_alpha = scalar_prod_logp0pw_beta_basis(p1, p0, DV, alpha)
            scalprods_ldv_alpha, scalprods_lscale = scalar_prod_logp0pw_beta_basis_npf(p1, p0, DV, alpha)

            f.root.embeddings_ldv.append(scalprods_ldv_alpha)
            f.root.embeddings_lscale.append(scalprods_lscale)

def limit_embeddings_ldv(dictionary, p_embeddings, p0, DV, outdirname, emb_name, point_name, chunk_size=10000, ntop=1, weighted=False, frac=False):
    # print("calculating limit vectors for alpha minus infinity")

    full_path = os.path.join(outdirname, get_alpha_ldv_name("limit", emb_name, point_name, ntop=ntop, weighted=weighted, frac=frac))

    #check if file already exists and is not empty (nothing to do)
    if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
        return

    all_words = list(dictionary.keys())

    with tables.open_file(full_path, 'w') as f:

        atom = tables.Atom.from_dtype(p_embeddings.dtype)

        vec_size = DV.shape[1]
        # 0 means the array can be expanded on that axis
        array_c = f.create_earray(f.root, 'embeddings_ldv', atom, (0, vec_size))

        for i in range(0, len(all_words), chunk_size):
            words = all_words[i: i + chunk_size]

            # proj logmap
            p1 = get_word_embeddings(words, p_embeddings, dictionary)

            if frac:
                pf = p1/p0.reshape(1,-1)
            else:
                pf = p1

            idxs_star = np.argpartition(-pf, kth=ntop, axis=1)[:, :ntop]

            # limit_ldv = DV[idxs_star].sum(axis=1)
            weights = 1
            if weighted:
                weights = np.zeros(idxs_star.shape)

                for i in range(idxs_star.shape[0]):
                    for j in range(idxs_star.shape[1]):
                        weights[i, j] = pf[i, idxs_star[i, j]]

                weights = np.expand_dims(weights, axis=-1)

            limit_ldv = np.sum(DV[idxs_star] * weights, axis=1)

            # old code only for the max, only for LE
            # chi_star = np.argmax(p1, axis=1)
            # limit_ldv = DV[chi_star]

            f.root.embeddings_ldv.append(limit_ldv)


def write_to_file(fstream, words, embeddings):
    for w, emb in zip (words, embeddings):
        line = " ".join(map(str, emb))
        line = w+" "+line+"\n"
        fstream.write(line)
    fstream.flush()



dsdir = '/ssd_data/text/cooccurrences/'
# simplewiki_sw6_fnamecc = dsdir + 'simplewiki201711/simplewiki201711-sw6-cooccurrence.bin'
# simplewiki_sw10_fnamecc = dsdir + 'simplewiki201711/simplewiki201711-sw10-cooccurrence.bin'
simplewiki_fnamevc = dsdir + 'simplewiki201711/simplewiki201711-vocab.txt'

# enwiki_sw10_fnamecc = dsdir + 'enwiki201710/enwiki201710-sw10-cooccurrence.bin'
enwiki_fnamevc = dsdir + 'enwiki201710/enwiki201710-vocab.txt'

geb_fnamevc = dsdir + 'geb/geb-vocab.txt'

fnamevc_dict = {
    "swiki": simplewiki_fnamevc,
    "enwiki": enwiki_fnamevc,
    "geb": geb_fnamevc
}


def make_all(corpus, vecsize, nepoch, alphas, baseoutdir, exp_name_for_dir):
    g_reader = readers.get_reader("glove")

    gname = corpus + get_suffix(vecsize, nepoch)
    outdirname = os.path.join(baseoutdir, corpus + exp_name_for_dir + "/" + gname)

    os.makedirs(outdirname, exist_ok=True)


    # select which vocabulary and cooccurrence file to use
    fnamevc = fnamevc_dict[corpus]
    (dictionary_size, v_dictionary, v_reversed_dictionary) = g_reader.read_dictionary(fnamevc)
    word_counts = g_reader.read_word_counts(fnamevc)
    pud = word_counts / np.sum(word_counts)


    g_dict, g_vecs, g_tuple = load_glove(corpus, vecsize, nepoch, calc_prob=False)

    # g_p_w, g_p_cIw = g_tuple
    U = g_vecs["u"][:-1, :]
    V = g_vecs["v"][:-1, :]

    calculate_all_embeddings(U, V, pud, alphas, v_dictionary, outdirname)



parser = argparse.ArgumentParser(description='Similarities logmaps and alphas.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('corpus', choices=["swiki", "enwiki", "geb"], help='Corpus for which to evaluate similarities.')
parser.add_argument('--outputdir', '-o', help='outputdir')
parser.add_argument('--sumvecs', action="store_true", help='Use the U+V instead of U as parameters for the exp[onential family.')
parser.add_argument('-v', '--vecsizes', nargs='+', type=int, help='the size of glove to load.')
parser.add_argument('-n', '--nepochs', nargs='+', type=int, help='the training epoch of glove to load.')

args = parser.parse_args()
corpus = args.corpus

sumvecs = args.sumvecs

exp_name_for_dir = "-alpha-emb"

DIGITS_ALPHA = 2
array_of_ranges = [np.arange(-10., 10.1, 0.2),
                   np.arange(-2, 2, 0.05),
                   np.arange(-30, -10, 1)]

alphas = np.sort(list(set(map(partial(round, ndigits=DIGITS_ALPHA),
                            np.concatenate(array_of_ranges)))))

# solve the ambiguity, 0, -0
alphas = np.array([a if not np.isclose(a, 0.) else 0. for a in alphas])

baseoutdir = args.outputdir

vecsizes = args.vecsizes
nepochs = args.nepochs
# vecsize = 300 #[100, 200, 300, 400]:
# nepoch = 1000 #[200, 400, 600, 800, 1000]:
for vecsize, nepoch in product(vecsizes, nepochs):
    make_all(corpus, vecsize, nepoch, alphas, baseoutdir, exp_name_for_dir)


