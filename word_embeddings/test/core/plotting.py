import matplotlib.pyplot as plt
import pickle
import numpy as np

def initialize_plot(title, xlabel=None, ylabel=None, figsize=None):
    figure = plt.figure(figsize=figsize)
    axes = figure.add_subplot(1, 1, 1)
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    return axes

def finalize_plot(axes, outname, xlim=None, ylim=None, legendloc=None):
    if xlim:
        axes.set_xlim(xlim[0], xlim[1])
    if ylim:
        axes.set_ylim(ylim[0], ylim[1])

    bbox_extra_artists = None
    if legendloc=="outside":
        lgd = axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        bbox_extra_artists = [lgd]
    else:
        axes.legend(loc=legendloc)

    plt.savefig(outname, bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')

def save_object(obj, outname):
    with open(outname,'wb') as outstream:
        pickle.dump(obj, outstream)

def load_data_dict(dataname):
    with open(dataname, 'br') as datastream:
        data_dict = pickle.load(datastream)
    return data_dict


def merge_data_dicts(data_dict_list):
    methods = np.array([ddc['plot_method'] for ddc in data_dict_list])
    plot_method_name = methods[0]
    if not np.all(methods == plot_method_name):
        raise ValueError("the plot specifications in the different input files must have all the same plot_method.")

    xlabels = np.array([ddc['x-label'] for ddc in data_dict_list])
    xlabel = xlabels[0]
    if not np.all(xlabels == xlabel):
        raise ValueError(
            "the plot specifications in the different input files must have the same data types (x-labels did not match).")

    ylabels = np.array([ddc['y-label'] for ddc in data_dict_list])
    ylabel = ylabels[0]
    if not np.all(ylabels == ylabel):
        raise ValueError(
            "the plot specifications in the different input files must have the same data types (y-labels did not match).")

    data_dict_merged = data_dict_list[0]
    for i in range(1, len(data_dict_list)):
        data_dict_merged['data'].append(data_dict_list[i]['data'])

    return data_dict_merged


def plot_data(data, axes, plot_method, plot_args=[], plot_kwargs={}):
    plot_method = getattr(plt, plot_method)
    for dc in data:
        plot_method(dc['x'], dc['y'], *plot_args, label=dc['label'], **plot_kwargs)


def create_plot_data_dict(xs_list, ys_list, label_list, x_label, y_label, plot_method_name="plot"):
    data = [{'x': xs, 'y': ys, 'label': label} \
            for (xs, ys, label) in zip(xs_list, ys_list, label_list)]

    dict_tosave = {'plot_method': plot_method_name, 'x-label': x_label, 'y-label': y_label, 'data': data}
    return dict_tosave


def get_ok_methods(collection, ok_method):
    couples_methods = []
    for d in collection:
        for m in collection[d]:
            if ok_method(m):
                lm = "limit-" + m
                couples_methods.append((m, lm))
    couples_methods = list(set(couples_methods))
    methods, limit_methods = zip(*couples_methods)
    return methods, limit_methods


def get_ok_datasets(collection, ok_dataset):
    datasets = []
    for d in collection:
        if ok_dataset(d):
            datasets.append(d)
    return datasets


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

