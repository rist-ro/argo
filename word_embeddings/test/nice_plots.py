import os
import subprocess
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from core.plotting import get_ok_datasets, get_ok_methods, is_a_split, is_in_point, is_limit_method, is_method

matplotlib.rcParams.update({'savefig.dpi': '300'})

fontsize=40
fonttitlesize=48
fontaxeslabelsize=40
fontlegendsize=36
ticksize = 44

matplotlib.rcParams.update({'font.size': fontsize})
matplotlib.rcParams.update({'font.weight': 'bold'})
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

matplotlib.rcParams.update({'legend.frameon': False})
matplotlib.rcParams.update({'legend.fontsize': fontlegendsize})
matplotlib.rcParams.update({'legend.borderaxespad': 1.0})

matplotlib.rcParams.update({'lines.linewidth': 2.0})
matplotlib.rcParams.update({'lines.markersize': 9})

matplotlib.rcParams.update({'axes.titlesize': fonttitlesize})
matplotlib.rcParams.update({'axes.titleweight': 'bold'})
matplotlib.rcParams.update({'axes.labelsize': fontaxeslabelsize})
matplotlib.rcParams.update({'axes.labelweight': 'bold'})

matplotlib.rcParams.update({'xtick.labelsize': ticksize})
matplotlib.rcParams.update({'ytick.labelsize': ticksize})

# matplotlib.rcParams.update({'axes.labelpad': 16.0})
# matplotlib.rcParams.update({'xtick.major.pad': 10.0})
# matplotlib.rcParams.update({'ytick.major.pad': 5.0})

figsize = matplotlib.rcParams['figure.figsize']
figsize[0] = 15
figsize[1] = 15

import numpy as np
import pickle
from mylogging import init_stream_logger, init_logger

stream_logger = init_stream_logger()


def load_similarities(simdir):
    with open(simdir + "/base-similarities.pkl", 'rb') as f:
        base_similarities = pickle.load(f)

    with open(simdir + "/alpha-similarities.pkl", 'rb') as f:
        alpha_similarities = pickle.load(f)

    return base_similarities, alpha_similarities


def load_analogies(analogdir):
    with open(analogdir + "/base-analogies.pkl", 'rb') as f:
        base_analogies = pickle.load(f)

    with open(analogdir + "/alpha-analogies.pkl", 'rb') as f:
        alpha_analogies = pickle.load(f)
    #     alpha_analogies = None

    return base_analogies, alpha_analogies


def load_sim_an(host, corpus_id):
    corpus, vstr, nstr = corpus_id.split('-')
    vecsize = int(vstr[1:])
    nepoch = int(nstr[1:])

    simbasedir = "/data/{:}_hd1/text/new-similarities".format(host)
    analogbasedir = "/data/{:}_hd1/text/new-analogies".format(host)
    embbasedir = "/data/{:}_hd1/text/new-alpha-embeddings/{:}-alpha-emb".format(host, corpus)

    embdir = os.path.join(embbasedir, corpus_id)
    simdir = os.path.join(simbasedir, corpus_id)
    analogdir = os.path.join(analogbasedir, corpus_id)

    alphas = np.load(embdir + "/alphas.npy")
    base_similarities, alpha_similarities = load_similarities(simdir)
    base_analogies, alpha_analogies = load_analogies(analogdir)

    similarities = {
        "base" : base_similarities,
        "alpha" : alpha_similarities
    }

    analogies = {
        "base" : base_analogies,
        "alpha" : alpha_analogies
    }

    return corpus, vecsize, nepoch, alphas, similarities, analogies

# GEB
geb_host = "thor"
geb_id = "geb-v300-n1000"

# ENWIKI
enwiki_host = "thor"
enwiki_id = "enwiki-v300-n1000"

outputdir = '/data2/text/nice_figures_new'

geb_corpus, geb_vecsize, geb_nepoch, geb_alphas, geb_similarities, geb_analogies = load_sim_an(geb_host, geb_id)

enwiki_corpus, enwiki_vecsize, enwiki_nepoch, enwiki_alphas, enwiki_similarities, enwiki_analogies = load_sim_an(enwiki_host, enwiki_id)

np.testing.assert_array_equal(geb_alphas, enwiki_alphas)
alphas = geb_alphas
amin = min(alphas)
amax = max(alphas)

# SIMILARITIES
base_methods_sim = {
    # 'p_cIw-cn' : ('cyan', '-'),
    'wikigiga5-u+v' : ('purple', '-'),
    'u-cn' : ('k', '--'),
    'u+v-cn' : ('k', '-'),
}

base_methods_an = {
    'wikigiga5-u+v' : ('purple', '-'),
    'u-cn' : ('k', '--'),
    'u+v-cn' : ('k', '-'),
}

to_replace = [('-nI-I', '-I'), ('-nF-F', '-F')]

def nice_label(m):
    try:
        nl = nice_label_base[m]
    except:
        if is_method(m, 'u'):
            nl = "E-U-"+m.split("u-plog-")[1]
        elif is_limit_method(m, 'u'):
            nl = "LE-U-"+m.split("limit-u-plog-")[1]
        elif is_method(m, 'u+v'):
            nl = "E-U+V-"+m.split("u+v-plog-")[1]
        elif is_limit_method(m, 'u+v'):
            nl = "LE-U+V-"+m.split("limit-u+v-plog-")[1]
        else:
            raise Exception("unexpected method name {:}".format(m))

        for s1, s2 in to_replace:
            nl = nl.replace(s1, s2)

    return nl

nice_label_base = {
    "p_cIw" : "p_data",
    "p_cIw-cn" : "p_data-cn",
    "wikigiga5-u+v" : "WG5-U+V",
    "wikigiga5-u+v-cn" : "WG5-U+V-cn",
    "u+v-cn" : "U+V-cn",
    "u-cn": "U-cn"
}


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
from cycler import cycler
from itertools import cycle
cc = (cycler(color=list(colors)))


def plot_base(base_methods, base_results, axis):
    for bm in base_methods:
        c, s = base_methods[bm]
        value = base_results[bm]
        axis.hlines(value, amin, amax, linestyles=s, color=c, label=nice_label(bm))


def plot_all(enwiki_collections, geb_collections, datasets, methods, limit_methods, base_methods, plot_folder, get_ylim):
    titles = ["enwiki", "geb"]

    for d in datasets:
        fig, axs = plt.subplots(nrows=1, ncols=2, sharex='all')
        # fig.suptitle(d)
        for i, ax in enumerate(axs):
            ax.set_title(titles[i])
            ax.set_ylim(get_ylim(d))

        cc = cycle(colors)
        for m,lm in zip(methods, limit_methods):
            enwiki_curve = enwiki_collections['alpha'][d][m]
            enwiki_lim = enwiki_collections['alpha'][d][lm]
            geb_curve = geb_collections['alpha'][d][m]
            geb_lim = geb_collections['alpha'][d][lm]

            c = next(cc)

            axs[0].plot(alphas, enwiki_curve, color=c, label=nice_label(m))
            axs[0].hlines(enwiki_lim, amin, amax, linestyles='--', color=c)
            axs[0].set_xlim(-10,6)
            #axs[0].grid()

            axs[0].set_xticks(np.arange(-10, 7, 2))
            minor_locator = AutoMinorLocator(2)
            axs[0].xaxis.set_minor_locator(minor_locator)
            
            axs[0].set_yticks(np.arange(min(get_ylim(d)), max(get_ylim(d))+1, 2))
            minor_locator = AutoMinorLocator(2)
            axs[0].yaxis.set_minor_locator(minor_locator)
            
            axs[0].set_axisbelow(True)
            #ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
            #ax.grid(which='minor', linestyle='-', linewidth='0.5', color='gray')
            axs[0].grid(which='major', linestyle='-', linewidth='1', color='lightgray')
            axs[0].grid(which='minor', linestyle='--', linewidth='1', color='lightgray')

            axs[1].plot(alphas, geb_curve, color=c, label=nice_label(m))
            axs[1].hlines(geb_lim, amin, amax, linestyles='--', color=c)
            axs[1].set_xlim(-10,7)
            axs[1].grid()

            axs[1].set_xticks(np.arange(-10, 7, 2))
            minor_locator = AutoMinorLocator(2)
            axs[1].xaxis.set_minor_locator(minor_locator)
            
            axs[1].set_yticks(np.arange(min(get_ylim(d)), max(get_ylim(d))+1, 2))
            minor_locator = AutoMinorLocator(2)
            axs[1].yaxis.set_minor_locator(minor_locator)
            
            axs[1].set_axisbelow(True)
            #ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
            #ax.grid(which='minor', linestyle='-', linewidth='0.5', color='gray')
            axs[1].grid(which='major', linestyle='-', linewidth='1', color='lightgray')
            axs[1].grid(which='minor', linestyle='--', linewidth='1', color='lightgray')

        plot_base(base_methods, enwiki_collections['base'][d], axs[0])
        plot_base(base_methods, geb_collections['base'][d], axs[1])
        lgd = axs[0].legend(loc = 'upper center', bbox_to_anchor = (1, 0.0), ncol=4)
        #plt.tight_layout()
        fig.subplots_adjust(bottom=0.4) # or whatever
        path = plot_folder+"/"+d+".png"
        plt.savefig(path) #, bbox_extra_artist = [lgd])
        print("saved " + path)
        plt.close()
        subprocess.run(["convert", "-trim", path, path])




thetas = ["u+v"]
points = ["0", "u", "ud"]
plot_sims_folder = os.path.join(outputdir, "similarities")
os.makedirs(plot_sims_folder, exist_ok=True)

def ok_method(method_str):
    ok_theta = np.any([is_method(method_str, t) for t in thetas])
    ok_point = np.any([is_in_point(method_str, p) for p in points])
    ok_norm = not ("-cn" in method_str)
    return ok_theta and ok_point and ok_norm

def ok_dataset(dataset_str):
    return not is_a_split(dataset_str)

datasets = get_ok_datasets(enwiki_similarities['alpha'], ok_dataset)
methods, limit_methods = get_ok_methods(enwiki_similarities['alpha'], ok_method)
methods, limit_methods = zip(*sorted(zip(methods, limit_methods)))

def get_sims_ylim(d):
    if "all" in d:
        return (50, 68)
    if "mc" in d:
        return (60, 90)
    if "wordsim" in d:
        return (50, 85)
    if "rw" in d:
        return (30, 60)
    if "scws" in d:
        return (40, 70)
    if "simlex" in d:
        return (20, 50)
    if "mturk" in d:
        return (45, 75)
    if "rg" in d:
        return (60, 90)
    if "men" in d:
        return (50, 80)


plot_all(enwiki_similarities, geb_similarities, datasets, methods, limit_methods, base_methods_sim, plot_sims_folder, get_sims_ylim)


def get_an_ylim(d):
    if "tot" in d:
        return (62, 78)
    if "sem" in d:
        return (60, 90)
    if "syn" in d:
        return (45, 75)


thetas = ["u+v"]
points = ["0", "u", "ud"]
plot_an_folder = os.path.join(outputdir, "analogies")
os.makedirs(plot_an_folder, exist_ok=True)

def ok_method(method_str):
    ok_theta = np.any([is_method(method_str, t) for t in thetas])
    ok_point = np.any([is_in_point(method_str, p) for p in points])
    ok_norm = ("-n" in method_str)

    if "-n" in method_str:
        norm_str, prod_str = method_str.split('-')[-2:]
        ok_prod = (norm_str[1]==prod_str)

    return ok_theta and ok_point and ok_norm and ok_prod


datasets = get_ok_datasets(enwiki_analogies['alpha'], ok_dataset)
methods, limit_methods = get_ok_methods(enwiki_analogies['alpha'], ok_method)
methods, limit_methods = zip(*sorted(zip(methods, limit_methods)))

plot_all(enwiki_analogies, geb_analogies, datasets, methods, limit_methods, base_methods_an, plot_an_folder, get_an_ylim)

