import argparse
import os
from matplotlib import pyplot as plt
from core.load_embeddings import load_all_emb_base
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


parser = argparse.ArgumentParser()
parser.add_argument('embid', type=str, help="embid of the embeddings to plot")
parser.add_argument('--outputdir', '-o', help='outputdir')

args = parser.parse_args()

wemb_id = args.embid # "enwiki-v300-n1000"

simbasedir = "/data1/text/new-similarities"
analogbasedir = "/data1/text/new-analogies"
embprebasedir = "/data1/text/new-alpha-embeddings"


corpus, vstr, nstr = wemb_id.split('-')
vecsize = int(vstr[1:])
nepoch = int(nstr[1:])

embbasedir = os.path.join(embprebasedir, corpus+"-alpha-emb")

embdir = os.path.join(embbasedir, wemb_id)
simdir = os.path.join(simbasedir, wemb_id)
analogdir = os.path.join(analogbasedir, wemb_id)

baseoutputdir = args.outputdir
plot_sims_folder = os.path.join(baseoutputdir, "figures", "similarities", wemb_id)
os.makedirs(plot_sims_folder, exist_ok=True)

plot_an_folder = os.path.join(baseoutputdir, "figures", "analogies", wemb_id)
os.makedirs(plot_an_folder, exist_ok=True)

alphas, I0, Iu, Iud, Ius = load_all_emb_base(embdir)
base_similarities, alpha_similarities = load_similarities(simdir)
base_analogies, alpha_analogies = load_analogies(analogdir)

import ipdb; ipdb.set_trace()

I0_inv = np.linalg.inv(I0)
Iu_inv = np.linalg.inv(Iu)
Iud_inv = np.linalg.inv(Iud)
Ius_inv = np.linalg.inv(Ius)

# SIMILARITIES


base_methods_sim = {
    'p_cIw-cn' : ('m', '-'),
    'wikigiga5-u+v' : ('orange', '-'),
    'u-cn' : ('k', '--'),
    'u+v-cn' : ('k', '-'),
}

base_methods_an = {
    'wikigiga5-u+v' : ('orange', '-'),
    'u-cn' : ('k', '--'),
    'u+v-cn' : ('k', '-'),
}

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
from cycler import cycler
cc = (cycler(color=list(colors)))
it = iter(cc)
from itertools import cycle


amin = min(alphas)
amax = max(alphas)

def get_ok_methods(collection, ok_method):
    couples_methods = []
    for d in collection:
        for m in collection[d]:
            if ok_method(m):
                lm = "limit-" + m
                couples_methods.append((m, lm))
    couples_methods = list(set(couples_methods))
    methods, limit_methods = zip(*sorted(couples_methods))
    return methods, limit_methods


def get_ok_datasets(collection, ok_dataset):
    datasets = []
    for d in collection:
        if ok_dataset(d):
            datasets.append(d)
    return sorted(datasets)


def is_method(method_str, theta):
    #     'u-plog-ud-cnI-I'
    return method_str.startswith("{:}-plog".format(theta))


def is_limit_method(method_str, theta):
    #     'limit-u-plog-ud-cnI-I'
    return method_str.startswith("limit-{:}-plog".format(theta))


def is_in_point(method_str, point_name):
    #     'u-plog-ud-cnI-I'
    found_pname = method_str.split("-plog-")[1].split("-")[0]
    return found_pname == point_name


def is_a_split(dataset_str):
    arr = dataset_str.split("-split_")
    l = len(arr)
    if l == 1:
        return False
    elif l == 2:
        return True
    else:
        raise Exception("what is this? Unexpected dataset: {:}".format(dataset_str))



thetas = ["u+v"]
points = ["0", "u", "ud"]


def ok_method(method_str):
    ok_theta = np.any([is_method(method_str, t) for t in thetas])
    ok_point = np.any([is_in_point(method_str, p) for p in points])
    ok_norm = not ("-cn" in method_str)
    return ok_theta and ok_point and ok_norm

def ok_dataset(dataset_str):
    return not is_a_split(dataset_str)

datasets = get_ok_datasets(alpha_similarities, ok_dataset)
methods, limit_methods = get_ok_methods(alpha_similarities, ok_method)

def get_sims_ylim(d):
    if "all" in d:
        return (40, 70)
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


def plot_base(base_methods, base_results):
    for bm in base_methods:
        c, s = base_methods[bm]
        value = base_results[bm]
        plt.hlines(value, amin, amax, linestyles=s, color=c, label=bm)


def plot_all(datasets, base_results, alpha_results, base_methods, methods, limit_methods, plot_folder, get_ylim):

    for d in datasets:
        fig = plt.figure(figsize=(15, 10))
        plt.title(d)

        cc = cycle(colors)

        for m,lm in zip(methods, limit_methods):
            curve = alpha_results[d][m]
            lim = alpha_results[d][lm]
            c = next(cc)
            plt.plot(alphas, curve, color=c, label=m)
            plt.hlines(lim, amin, amax, linestyles='--', color=c)

        plt.ylim(get_ylim(d))
        plot_base(base_methods, base_results[d])
        plt.legend()
        plt.tight_layout()
        path = plot_folder+"/"+d+".png"
        plt.savefig(path)
        plt.close()



plot_all(datasets, base_similarities, alpha_similarities, base_methods_sim, methods, limit_methods, plot_sims_folder, get_sims_ylim)


def get_an_ylim(d):
    if "sem" in d:
        return (60, 90)
    if "syn" in d:
        return (45, 75)
    if "tot" in d:
        return (50, 80)

thetas = ["u+v"]
points = ["0", "u"]#, "ud"]

def ok_method(method_str):
    ok_theta = np.any([is_method(method_str, t) for t in thetas])
    ok_point = np.any([is_in_point(method_str, p) for p in points])
    ok_norm = ("-n" in method_str)
    return ok_theta and ok_point and ok_norm


datasets = get_ok_datasets(alpha_analogies, ok_dataset)
methods, limit_methods = get_ok_methods(alpha_analogies, ok_method)

plot_all(datasets, base_analogies, alpha_analogies, base_methods_an, methods, limit_methods, plot_an_folder, get_an_ylim)


# TABLES

thetas = ["u", "u+v"]
points = ["0", "u", "ud"]


def ok_method(method_str):
    ok_theta = np.any([is_method(method_str, t) for t in thetas])
    ok_point = np.any([is_in_point(method_str, p) for p in points])
    ok_norm = not ("-cn" in method_str)
    return ok_theta and ok_point and ok_norm


def ok_dataset(dataset_str):
    return is_a_split(dataset_str)


datasets = get_ok_datasets(alpha_analogies, ok_dataset)
methods, limit_methods = get_ok_methods(alpha_analogies, ok_method)


def get_other_ds(d, s, datasets):
    other_ds = []
    for ods in datasets:
        od, os = ods.split("-split_")
        if od == d and s != os:
            other_ds.append(ods)
    return other_ds


cross_val = {}

for ds in datasets:
    cross_val[ds] = {}
    d, s = ds.split("-split_")
    for m in methods:
        curve = alpha_analogies[ds][m]

        i_star = np.argmax(curve)
        alpha_star = alphas[i_star]
        other_ds = get_other_ds(d, s, datasets)
        acc_star = np.mean([alpha_analogies[ods][m][i_star] for ods in other_ds])
        cross_val[ds][m] = (i_star, alpha_star, acc_star)


methods = cross_val["tot-split_0"].keys()
datasets = cross_val.keys()

accuracy_d = {}
for m in methods:
    for ds in datasets:
        d, s = ds.split("-split_")
        if accuracy_d.get(d, None) is None:
            accuracy_d[d] = {}
        if accuracy_d[d].get(m, None) is None:
            accuracy_d[d][m] = []

        istar, astar, accstar = cross_val[ds][m]
        accuracy_d[d][m].append([astar, accstar])

thetas = ["u+v"]
points = ["0", "u"]


def ok_method(method_str):
    ok_theta = np.any([is_method(method_str, t) for t in thetas])
    ok_point = np.any([is_in_point(method_str, p) for p in points])
    ok_norm = ("-n" in method_str)
    return ok_theta and ok_point and ok_norm


def ok_dataset(dataset_str):
    return not is_a_split(dataset_str)


output_table_file = plot_an_folder+"/cross_val_table.txt"

outstream = open(output_table_file, "w")

tab_datasets = get_ok_datasets(alpha_analogies, ok_dataset)
tab_methods, _ = get_ok_methods(alpha_analogies, ok_method)
tab_methods = sorted(tab_methods)

for d in tab_datasets:
    outstream.write("\hline\n")
    outstream.write("\multirow{19}{*}{" + d + "}\n")
    for m in tab_methods:
        cross_alphas, cross_accs = zip(*accuracy_d[d][m])
        a_m = np.mean(cross_alphas)
        a_std = np.std(cross_alphas)
        acc_m = np.mean(cross_accs)
        acc_std = np.std(cross_accs)
        outstream.write("  & {:} & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ \\\\\n".format(m, a_m, a_std, acc_m, acc_std))

for d in tab_datasets:
    outstream.write("\hline\n")
    outstream.write("\multirow{8}{*}{" + d + "}\n")
    i = 0
    for m in tab_methods:
        lm = "limit-" + m
        acc_m = alpha_analogies[d][lm][0]
        if i == 0:
            outstream.write("  & {:} & \multirow{{8}}{{*}}{{-}} & {:.2f} \\\\\n".format(lm, acc_m))
        else:
            outstream.write("  & {:} &  & {:.2f} \\\\\n".format(lm, acc_m))
        i += 1

for d in tab_datasets:
    outstream.write("\hline\n")
    outstream.write("\multirow{12}{*}{" + d + "}\n")
    i = 0
    for bm in base_methods_an:
        acc_m = base_analogies[d][bm][0]
        #         if i==0:
        #             print("  & {:} & \multirow{{12}}{{*}}{{-}} & {:.2f} \\\\".format(bm, acc_m))
        #         else:
        outstream.write("  & {:} &  & {:.2f} \\\\\n".format(bm, acc_m))
        i += 1

outstream.close()

# SIMILARITY TABLE

thetas = ["u", "u+v"]
points = ["0", "u", "ud"]

def ok_method(method_str):
    ok_theta = np.any([is_method(method_str, t) for t in thetas])
    ok_point = np.any([is_in_point(method_str, p) for p in points])
    ok_norm = not ("-cn" in method_str)
    return ok_theta and ok_point and ok_norm


def ok_dataset(dataset_str):
    return not is_a_split(dataset_str)


output_table_file = plot_sims_folder+"/sims_table.txt"

outstream = open(output_table_file, "w")
# tab_datasets = get_ok_datasets(alpha_similarities, ok_dataset)
tab_datasets = ['wordsim353', 'mc', 'rg', 'scws', 'wordsim353sim', 'wordsim353rel', 'men', 'mturk287', 'rw', 'simlex999', 'all']

tab_methods, _ = get_ok_methods(alpha_similarities, ok_method)
tab_methods = sorted(tab_methods)

lines = {}
header = "corpus & method & " + " & ".join(tab_datasets)
outstream.write(header+"\n")

tab_limit_methods = ["limit-"+m for m in tab_methods]

short = {
    "p_cIw" : "p_data",
    "wikigiga5-u+v" : "WG5-U+V",
    "u+v-cn" : "U+V",
    "u-cn": "U",
    "limit-u-plog-0-F" : "LE-U-0-F",
    "limit-u-plog-0-I": "LE-U-0-I",
    "limit-u-plog-u-F": "LE-U-u-F",
    "limit-u-plog-u-I": "LE-U-u-I",
    "limit-u-plog-ud-F": "LE-U-ud-F",
    "limit-u-plog-ud-I": "LE-U-ud-I",
    "limit-u+v-plog-0-F": "LE-U+V-0-F",
    "limit-u+v-plog-0-I": "LE-U+V-0-I",
    "limit-u+v-plog-u-F": "LE-U+V-u-F",
    "limit-u+v-plog-u-I": "LE-U+V-u-I",
    "limit-u+v-plog-ud-F": "LE-U+V-ud-F",
    "limit-u+v-plog-ud-I": "LE-U+V-ud-I",
}

for lm in tab_limit_methods:
    line = "& "+short[lm]
    for d in tab_datasets:
        line+= "& {:.2f} ".format(alpha_similarities[d][lm][0])
    line+= "\\\\\n"
    outstream.write(line)

# import pdb; pdb.set_trace()

for bm in base_methods_an:
    line = "& " + short[bm]

    for d in tab_datasets:
        bsim = base_similarities[d][bm][0]
        line += "& {:.2f} ".format(bsim)

    line += "\\\\\n"
    outstream.write(line)

outstream.close()
