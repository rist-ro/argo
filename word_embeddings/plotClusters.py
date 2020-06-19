import os
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
from glob import glob
from matplotlib.ticker import AutoMinorLocator

matplotlib.rcParams.update({'savefig.dpi': '300'})

fontsize=40
fonttitlesize=48
fontaxeslabelsize=40
fontlegendsize=36
ticksize = 44

# # 300
# ylims = {
#     'ap' : (50, 73),
#     'battig' : (40, 70),
#     'bless' : (70, 90),
#     'capitals' : (70, 100),
#     'essli-2008' : (40, 70),
# }

# 50
ylims = {
    'ap' : (45, 68),
    'battig' : (40, 70),
    'bless' : (65, 85),
    'capitals' : (70, 100),
    'essli-2008' : (40, 70),
}

emb_dirs = ["geb-v50-n1000"] #["geb-v50-n1000", "geb-v300-n1000"]
task_dirs = ['ap', 'battig', 'bless', 'capitals', 'essli-2008']


matplotlib.rcParams.update({'font.size': fontsize})
matplotlib.rcParams.update({'font.weight': 'bold'})
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

matplotlib.rcParams.update({'legend.frameon': False})
matplotlib.rcParams.update({'legend.fontsize': fontlegendsize})
matplotlib.rcParams.update({'legend.borderaxespad': 1.0})

matplotlib.rcParams.update({'lines.linewidth': 4.0})
matplotlib.rcParams.update({'lines.markersize': 9})

matplotlib.rcParams.update({'axes.titlesize': fonttitlesize})
matplotlib.rcParams.update({'axes.titleweight': 'bold'})
matplotlib.rcParams.update({'axes.labelsize': fontaxeslabelsize})
matplotlib.rcParams.update({'axes.labelweight': 'bold'})

matplotlib.rcParams.update({'xtick.labelsize': ticksize})
matplotlib.rcParams.update({'ytick.labelsize': ticksize})

figsize = matplotlib.rcParams['figure.figsize']
figsize[0] = 15 #6.4*2
figsize[1] = 15 #4.8*2

pd_csv_kwargs = {
            "sep" : " "
        }


# prop_cycle = plt.rcParams['axes.prop_cycle']
# colors = prop_cycle.by_key()['color']
# from cycler import cycler
# from itertools import cycle
# cc = (cycler(color=list(colors)))

from scipy.interpolate import make_interp_spline
import numpy as np

def plot_single_errorbar(df, x, y, string_replace={}, **kwargs):
    xdata = []
    ydata = []
    yerr = []

    for label, df in df.groupby(x):
        try:
            x_value = string_replace[label]
        except:
            x_value = float(label)

        xdata.append(x_value)
        # ydata.append(df.mean()[y])
        ydata.append(df.max()[y])
        yerr.append(df.std()[y])


    xdata, ydata, yerr = zip(*sorted(zip(xdata, ydata, yerr)))
    # plt.errorbar(xdata, ydata, yerr=yerr, **kwargs)
    xnew = np.linspace(np.min(xdata), np.max(xdata), 31)
    spl = make_interp_spline(xdata, ydata, k=1)
    ynew = spl(xnew)
    plt.plot(xnew, ynew*100, marker='o', **kwargs)
    # plt.plot(xdata, np.array(ydata)*100, marker='o', **kwargs)


thetas = ['U+V'] # ['U'] #['U+V']
def nice_label(filename):
    arr = filename.split("/")[-1].split('_')[2:-1]
    evsize = arr[0].split('-')[1].replace('v', 'E')
    theta = arr[1].upper()
    point = arr[2]
    norm = arr[3]
    prod = arr[4]

    skip = False
    if norm[1]!=prod:
        skip = True
    if not (theta in thetas):
        skip = True

    return "-".join([evsize, theta, point, prod]), skip


def log_clustering_name():
    "{:}_clustering".format(task_dir)

def _tryfloat(v):
    try:
        value = float(v)
    except ValueError:
        value = v

    return value


def add_one_plot(matches, x, y):
    replace_strings_for_plots = {
        'limit' : -6.,  # since for plotting I need to choose a value
    }

    for filename in matches:
        label, skip = nice_label(filename)
        if skip:
            continue

        df = pd.read_csv(filename, **pd_csv_kwargs)
        # import pdb; pdb.set_trace()
        plot_single_errorbar(df, x, y, string_replace=replace_strings_for_plots, alpha=0.9, label=label)


def finalize_plot(ylim, plotoutpath, suptitle=''):
    plt.title(suptitle)
    xlim = (-6., 4.)
    plt.ylim(ylim)
    plt.xlim(xlim)

    ax = plt.gca()
    ax.set_xticks(np.arange(xlim[0], xlim[1], 1))
    minor_locator = AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.set_yticks(np.arange(min(ylim), max(ylim) + 1, 2))
    minor_locator = AutoMinorLocator(2)
    ax.yaxis.set_minor_locator(minor_locator)

    ax.set_axisbelow(True)
    # ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
    # ax.grid(which='minor', linestyle='-', linewidth='0.5', color='gray')
    ax.grid(which='major', linestyle='--', linewidth='1', color='lightgray')
    ax.grid(which='minor', linestyle=':', linewidth='1', color='lightgray')

    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    plt.tight_layout()
    plt.savefig(plotoutpath)
    plt.close()



clusters_basedir = "/data/thor_hd2/text/clustering_norm/pca_clusters"
outdirname = os.path.join(clusters_basedir, "collected")
os.makedirs(outdirname, exist_ok=True)


for task_dir in task_dirs:
    for emb_dir in emb_dirs:
        path_list = [clusters_basedir, emb_dir, task_dir, "{:}_clustering_*_skm.txt".format(task_dir)]
        matches = glob(os.path.join(*path_list))
        add_one_plot(sorted(matches), 'alpha', 'purity')

    plotoutpath = os.path.join(outdirname, '{:}_skm_pur.png'.format(task_dir))
    finalize_plot(ylims[task_dir], plotoutpath, suptitle=task_dir)

