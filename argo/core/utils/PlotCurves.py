# it takes a lifetime to import this (Riccardo)
#from argo.core.utils.argo_utils import make_list, create_list_colors
from argo.core.utils.Curve import Curve
from .AbstractPlot import AbstractPlot, make_list, create_list_colors


from itertools import product

import re

import os

import numpy as np

# see https://matplotlib.org/api/markers_api.html
markers = "ov^<>spP*Dd"  # +x


class PlotCurves(AbstractPlot):

    def __init__(self):
        super(PlotCurves, self).__init__()

    def rows_columns(self, panels, folders_cartesian_fixed):
        n_columns = len(panels)
        n_rows = np.max([len(p) for p in panels])
        return n_rows, n_columns
        
    def plot_vertical_panel(self,
                            vertical_panels,
                            c,
                            n_rows,
                            n_columns,
                            legend,
                            fig,
                            folders_cartesian_fixed,
                            group_by_values,
                            source,
                            degree,
                            string_to_val):

        r = 0
        
        for panel in vertical_panels:
            
            ax = fig.add_subplot(n_rows, n_columns, r * n_columns + c + 1)

            if len(group_by_values) == 0:
                max_colors = len(folders_cartesian_fixed)
            else:
                max_colors = len(list(group_by_values.values())[0])

            if max_colors == 0:
                max_colors = 1
            list_colors = create_list_colors(max_colors)

            firstCurve = None

            for curve in panel:

                folders_cartesian_fixed = sorted(folders_cartesian_fixed)

                for marker, color, (source, directory) in self.get_colors_pairs(folders_cartesian_fixed,
                                                                      group_by_values):
                    label = self.create_label(curve, directory, legend, source)
                    cur = Curve(directory=directory, string_to_val=string_to_val, label=label, **curve)

                    if firstCurve is None:
                        firstCurve = cur

                    cur.create_plot(markers[marker % len(markers)], list_colors[color], ax, degree)


            # sometimes it raises an error, so I added a check (Luigi)
            if firstCurve is not None:
                firstCurve.set_panel_properties(ax)

            r += 1

def join_dirs_path(source, d):
    return d.replace(source + '/', '').replace('//', '/')  # .replace('/','-')

def get_value(directory, l):
    if len(l) == 2:
        (param, name) = l
        m = re.search('(-|^)' + param + '([\._A-Za-z0-9,]+)' + '(-|$)', directory)
        if m and m.group(2):
            return name + m.group(2)
        return ""  # "param not found"
    elif len(l) == 3:
        (param, i, name) = l
        splits = directory.split('/')
        #pdb.set_trace()
        m = re.search('(-|^)' + param + '([\._A-Za-z0-9,]+)' + '(-|$)', splits[i])
        if m and m.group(2):
            return name + m.group(2)
        return ""  # "param not found"
    elif len(l) == 4:
        (param, subparam, i, name) = l
        splits = directory.split('/')
        m = re.search('(-|^)' + param + '([\._A-Za-z0-9,]+)' + '(-|$)', splits[i])
        if m and m.group(2):
            found = m.group(2)
            m = re.search('(_)' + subparam + '([\.A-Za-z0-9]+)' + '(_|$)', found)
            if m and m.group(2):
                return name + m.group(2)
        return ""  # "param not found"
    else:
        raise Exception("The length of the token should be 2 or 3")

    return "param not found"


# # BEFORE ERAISING THIS WE NEED TO MERGE THE MODIFICATIONS DONE HERE. I was not able to do it quickly and I have no time right now.. (Riccardo)
# def plot_curves(panels, where, block, fixed_over, logs_directory, depth, dataset, tag, filename="log", dirname=".",
#                 figsize=(26, 10), legend=[], rcParams={}, string_to_val={}, small_size=8, medium_size=10, bigger_size=12, degree=10):
#
#     plt.rc('font', size=small_size)  # controls default text sizes
#     plt.rc('axes', titlesize=small_size)  # fontsize of the axes title
#     plt.rc('axes', labelsize=medium_size)  # fontsize of the x and y labels
#     plt.rc('xtick', labelsize=small_size)  # fontsize of the tick labels
#     plt.rc('ytick', labelsize=small_size)  # fontsize of the tick labels
#     plt.rc('legend', fontsize=small_size)  # legend fontsize
#     plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title
#     matplotlib.rcParams.update(rcParams)
#
#     source = logs_directory + "/" + dataset
#
#     dirs = []
#     for (parent, subdirs, files) in os.walk(source):
#         if len(join_dirs_path(source, parent).split('/')) == depth:
#             dirs.append(parent)
#
#     folders_where = [d for d in dirs if
#                      check_where(join_dirs_path(source, d), where) and check_block(join_dirs_path(source, d), block)]
#
#     fixed_over_values = collections.OrderedDict()
#     for f in fixed_over:
#         fixed_over_values[f] = []
#
#     for d in folders_where:
#
#         directory = join_dirs_path(source, d)
#         for f_tuple in fixed_over:
#             if not isinstance(f_tuple, tuple):
#                 f = f_tuple
#                 m = re.search('(-|^)' + f + '([\._A-Za-z0-9]+)' + '(-|$)', directory)
#                 if m:
#                     if m.group(2) not in fixed_over_values[f]:
#                         # here I may something like -cELBO_A1_A2-
#                         # and I want to keep only ELBO
#                         token = m.group(2).split("_")[0]
#                         if token not in fixed_over_values[f]:
#                             fixed_over_values[f].append(token)
#             else:
#                 (f, i) = f_tuple
#                 if isinstance(i, int):
#                     # i is a number and thus it specify the level in which I have to look for
#                     splits = directory.split('/')
#                     m = re.search('(-|^)' + f + '([\._A-Za-z0-9]+)' + '(-|$)', splits[i])
#                     if m:
#                         if m.group(2) not in fixed_over_values[(f, i)]:
#                             fixed_over_values[(f, i)].append(m.group(2))
#                 else:
#                     # i is a tag, I hadve to look for the value of the sub_tag i
#                     # inside the tag f
#                     m = re.search('(-|^)' + f + '([\._A-Za-z0-9]*)' + i + '([\.A-Za-z0-9]*)' + '([\._A-Za-z0-9]*)' + '(-|$)', directory)
#                     if m and m.group(3) not in fixed_over_values[(f, i)]:
#                         token = m.group(3)
#                         fixed_over_values[(f, i)].append(token)
#
#     print(fixed_over_values)
#
#     def my_items(i):
#         #return [(a[0][0], a[1], a[0][1]) if len(a) == 2 else a for a in i.items()]
#         return [(a[0][0], a[1], a[0][1]) if (len(a) == 2 and isinstance(a, list)) else a for a in i.items()]
#
#     cartesian_fixed = [collections.OrderedDict(zip(fixed_over_values.keys(), p)) for p in
#                        product(*map(make_list, fixed_over_values.values()))]
#     cartesian_fixed_list = [d for d in cartesian_fixed]
#
#     for c_fixed in cartesian_fixed_list:
#
#         folders_cartesian_fixed = [d for d in folders_where if
#                                    check_where(join_dirs_path(source, d), my_items(c_fixed))]
#
#         print("============================================")
#         print(c_fixed)
#         for k in folders_cartesian_fixed:
#             print(k)
#
#         n_columns = len(panels)
#         n_rows = np.max([len(p) for p in panels])
#
#         # create figure
#         fig = plt.figure(figsize=figsize)
#         fig.suptitle(dataset, y=0.995)  # , fontsize=10)
#
#         c = 0
#
#         for vertical_panels in panels:
#
#             r = 0
#
#             for panel in vertical_panels:
#
#                 n_plots = len(panels)
#                 ax = fig.add_subplot(n_rows, n_columns, r * n_columns + c + 1)
#
#                 max_colors = len(folders_cartesian_fixed)
#
#                 if max_colors == 0:
#                     max_colors = 1
#                 list_colors = create_list_colors(max_colors)
#
#                 for curve in panel:
#
#                     folders_cartesian_fixed = sorted(folders_cartesian_fixed)
#                     for color, directory in enumerate(folders_cartesian_fixed):
#
#                         # file_path = source  + "/" + directory + "/" + curve["filename"]
#                         file_path = directory + "/" + curve["filename"]
#
#                         try:
#                             # read data from file
#                             with open(file_path) as f:
#                                 data = f.read()
#
#                             data = data.split('\n')
#
#                             offset = 1
#                             data = data[offset:-1]
#
#                             separator = curve.get("separator", "\t")
#                             x = [str2float(row.lstrip().split(separator)[curve["x"]], string_to_val) for row in data]
#                             y = [str2float(row.lstrip().split(separator)[curve["y"]], string_to_val) for row in data]
#
#                             # see https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
#
#                             assert (len(y) > 1)
#
#                             linestyle = curve.get("linestyle", '-')
#
#                             # print(color)
#                             if curve.get("logscale_y", 0):
#                                 plot_function = ax.semilogy
#                             else:
#                                 plot_function = ax.plot
#
#                             if curve.get("logscale_x", 0):
#                                 ax.set_xscale('log')
#
#                             mark_every = curve.get("mark_every", 25)
#                             markevery = min(len(y), mark_every)  # int(len(y) / 20)
#
#                             #pdb.set_trace()
#                             if len(legend) > 0:
#                                 "Legend has to be in the format: [(param, name)], where param is an id in the name"
#                                 label = ""
#                                 for l in legend:
#                                     piece = get_value(join_dirs_path(source, directory), l)
#                                     label += piece + " "
#                             elif "y_label" in curve:
#                                 file_path = directory + '/' + curve.get("legend", "")
#                                 if "legend" in curve and os.path.isfile(file_path):
#                                     f = open(file_path, "r")
#                                     label = f.read()
#                                     if label[-1] == '\n':
#                                         label = label[:-1]
#                                 else:
#                                     label = join_dirs_path(source, directory).replace('/', '\n')
#                             else:
#                                 label = None
#
#                             linewidth = curve.get("linewidth", 2)
#                             markersize = curve.get("markersize", 10)
#
#                             if curve.get("smoothing", 0):
#                                 y_smooth = smoothTriangle(y, degree)  # window size 51, polynomial order 3
#                                 plot_function(x, y_smooth, linestyle, c=list_colors[color % len(list_colors)],
#                                               label=label, marker=markers[color % len(markers)], markevery=markevery,
#                                               markersize=markersize, linewidth=linewidth)
#                                 plot_function(x, y, linestyle, c=list_colors[color % len(list_colors)], alpha=0.3,
#                                               linewidth=linewidth)
#                             else:
#                                 plot_function(x, y, linestyle, c=list_colors[color % len(list_colors)], label=label,
#                                               marker=markers[color % len(markers)], markevery=markevery, markersize=markersize,
#                                               linewidth=linewidth)
#
#                         except FileNotFoundError as e:
#                             print("FileNotFoundError cannot read file " + file_path + " " + str(e))
#                         except AssertionError as e:
#                             print("AssertionError file is empty " + file_path + " " + str(e))
#                         except ValueError as e:
#                             print("ValueError cannot read file " + file_path + " " + str(e))
#                         except IndexError as e:
#                             print("IndexError cannot read file " + file_path + " " + str(e))
#
#                 #first curve of the panel
#                 curve = panel[0]
#
#                 if "min_x" in curve:
#                     ax.set_xlim(left=curve["min_x"])
#                 else:
#                     ax.set_xlim(left=0)
#
#                 if "max_x" in curve:
#                     ax.set_xlim(right=curve["max_x"])
#                 if "min_y" in curve:
#                     ax.set_ylim(bottom=curve["min_y"])
#                 if "max_y" in curve:
#                     ax.set_ylim(top=curve["max_y"])
#
#                 ax.set_xlabel(curve.get("x_label", ''))
#                 if "y_label" in curve:
#                     ax.set_ylabel(curve["y_label"])
#
#                 loc = curve.get("legend_loc", "upper right")
#                 bbox_to_anchor = curve.get("bbox_to_anchor", None)
#                 if bbox_to_anchor is None:
#                     lgd = ax.legend(loc=loc)
#                 else:
#                     lgd = ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor)
#
#                 bbox_extra_artist_bool = curve.get("bbox_extra_artist", False)
#
#                 if curve.get("no_ticks", 0):
#                     xt = ax.get_xticks()
#                     # xt = np.append(xt, offset)
#                     xtl = xt.astype(int).tolist()
#                     # xtl[-1]= offset
#                     ax.set_xticks(xt)
#                     ax.set_xticklabels(xtl)
#
#                 ax.grid()
#
#                 r += 1
#
#             c += 1
#
#         plt.tight_layout()
#
#         def format_attr(k):
#             if len(k) == 1:
#                 return str(k)
#             else:
#                 if isinstance(k[1], int):
#                     return str(k[0]) + '[' + str(k[1]) + ']'
#                 else:
#                     return str(k[0]) + '_' + str(k[1])
#
#         attr = "-".join([format_attr(k) + str(v) for k, v in c_fixed.items()])
#         if len(attr) > 0:
#             attr = '_' + attr
#         print("saving: " + dirname + "/" + dataset + "_" + filename + "_" + tag + attr + ".png")
#         plot_kwargs = {
#             'bbox_inches' : 'tight',
#         }
#
#         if bbox_extra_artist_bool:
#             plot_kwargs.update({'bbox_extra_artists' : [lgd]})
#
#         plt.savefig(dirname + "/" + dataset + "_" + filename + "_" + tag + attr + ".png", **plot_kwargs)
#
#
# def str2float(s, string_to_val):
#     try:
#         val = string_to_val[s]
#     except:
#         val = float(s)
#
#     return val
#
#
#
# # see https://www.swharden.com/wp/2010-06-20-smoothing-window-data-averaging-in-python-moving-triangle-tecnique/
# def smoothTriangle(data, degree, dropVals=False):
#     """performs moving triangle smoothing with a variable degree."""
#     """note that if dropVals is False, output length will be identical
#     to input length, but with copies of data at the flanking regions"""
#     triangle = np.array(list(range(degree)) + [degree] + list(range(degree))[::-1]) + 1
#     smoothed = []
#     for i in range(degree, len(data) - degree * 2):
#         point = data[i:i + len(triangle)] * triangle
#         smoothed.append(sum(point) / sum(triangle))
#     if dropVals: return smoothed
#     smoothed = [smoothed[0]] * int(degree + degree / 2) + smoothed
#     while len(smoothed) < len(data): smoothed.append(smoothed[-1])
#
#     # some refinement
#     alphas = [alpha / degree / 2 for alpha in range(degree * 2)]
#     smoothed[:degree * 2] = [a * s + (1 - a) * d for a, s, d in zip(alphas, smoothed[:degree * 2], data[:degree * 2])]
#     return smoothed
