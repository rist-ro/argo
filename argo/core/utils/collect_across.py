import os, re
from glob import glob
from pprint import pprint
import pandas as pd
import numpy as np
from itertools import product

pd_csv_kwargs = {
            "sep" : "\t"
        }

def all_same(items):
    return all(x == items[0] for x in items)

def check_matches(matches, base_dir):
    trimmed_matches = [match.split(base_dir)[1].strip('/') for match in matches]
    ds_strings, net_strings = zip(*[m.split("/")[:2] for m in trimmed_matches])
    if not all_same(net_strings):
        raise Exception("net_strings are not all the same, check your regular expression, found: {:}".format(set(net_strings)))
    return sorted(ds_strings), list(set(net_strings))[0]

def get_ds_field_value(match, pre_ds_dir, post_ds_dir):
    value = match.partition(pre_ds_dir)[2].partition(post_ds_dir)[0]
    return value

def read_value(path, col, val, field):
    df = read_argo_csv(path, **pd_csv_kwargs)
    row = df[df[col]==val]
    return row[field].values[0]

def read_max_value(path, field, col_ref):
    df = read_argo_csv(path, **pd_csv_kwargs)
    imax = df[field].idxmax()
    return df[col_ref][imax], df[field][imax]

def read_argo_csv(path, **pd_csv_kwargs):
    df = pd.read_csv(path, **pd_csv_kwargs)
    df = df.rename(columns={"# epochs": "epoch"})
    rename_dict = {col: col.strip() for col in df.columns}
    df = df.rename(columns=rename_dict)
    return df

def convert_alpha(alpha_str, limitfloat):
    if alpha_str == 'limit':
        return limitfloat
    else:
        return float(alpha_str)

def tryfloat(alpha_str):
    try:
        value = float(alpha_str)
    except ValueError:
        value = alpha_str

    return value

def get_limit_float(data):
    found_limit = False
    xs = [tryfloat(d[0]) for d in data]
    if "limit" in xs:
        found_limit = True

    xs = [x for x in xs if isinstance(x, float)]

    return min(xs)-1, found_limit

def key_sort(a):
    if a[0] == 'limit':
        return -np.inf

    return a[0]


def collect_data_across_ds_single_before(base_dir, ds_dir, net_dir, log_file, outdirname="."):

    dir_list = [base_dir, ds_dir, net_dir, log_file]
    matches = glob(os.path.join(*dir_list))
    if len(matches)==0:
        raise ValueError("No file found matching the provided regexpr: `{:}`".format(os.path.join(*dir_list)))

    ds_strings, net_string = check_matches(matches, base_dir)
    pprint(ds_strings)

    pre_ds_dir, _, post_ds_dir = ds_dir.partition("*")
    ds_param_name = "alpha"
    log_name = os.path.splitext(log_file)[0]
    field_train = log_name+'_train'
    field_val = log_name+'_validation'
    field_test = log_name+'_test'

    all_fields = [ds_param_name, 'epoch', field_train, field_val, field_test]

    data = []
    for match in matches:
        x = get_ds_field_value(match, pre_ds_dir, post_ds_dir)
        epmax, y_val = read_max_value(match, field_val, 'epoch')
        y_train = read_value(match, 'epoch', epmax, field_train)
        y_test = read_value(match, 'epoch', epmax, field_test)
        data.append((tryfloat(x), epmax, y_train, y_val, y_test))

    sorted_data = sorted(data, key=key_sort)

    df = pd.DataFrame(sorted_data, columns=all_fields)

    outpath = os.path.join(outdirname, pre_ds_dir+'W'+post_ds_dir, net_string)
    os.makedirs(outpath, exist_ok=True)
    outpath = os.path.join(outpath, "collected_"+log_file)
    df.to_csv(outpath, index=False, float_format='%.6g', **pd_csv_kwargs)


def collect_data_across_ds_before(base_dir, ds_dirs, net_dirs, log_file, outdirname="."):
    for ds_dir, net_dir in product(ds_dirs, net_dirs):
        collect_data_across_ds_single_before(base_dir, ds_dir, net_dir, log_file, outdirname=outdirname)


# collect_dict = {
#     'main_field_spec' : ('a',),
#
#     'plot' : {
#         'main_field_value' : 'float',
#         'plot_field_spec' : ('alpha', ('a','v')),
#     },
#
#     'table' : {
#         'main_field_value' : 'limit',
#         'table_field_spec' : [
#             ('ntop', ('a', 't')),
#             ('weighted', ('a', 'w')),
#             ('fraction', ('a', 'f')),
#         ],
#     }
# }

def collect_data_across_ds_single(base_dir, ds_dir, net_dir, log_file, collect_dict, outdirname="."):
    log_filename = log_file['filename']
    target_col = log_file['target_col']

    dir_list = [base_dir, ds_dir, net_dir, log_filename]

    matches = glob(os.path.join(*dir_list))
    if len(matches)==0:
        raise ValueError("No file found matching the provided regexpr: `{:}`".format(os.path.join(*dir_list)))

    ds_strings, net_string = check_matches(matches, base_dir)
    pprint(ds_strings)

    mainfield_spec = collect_dict['main_field_spec']
    baseoutputname = name_with_wildcard(ds_dir, mainfield_spec)

    level = -3 # usually -3 for dataset in a split, basedir/dsdir/netdir/logfile
    # csv for plot

    conf_plot = collect_dict['plot']
    conf_table = collect_dict['table']

    # log_name = os.path.splitext(os.path.basename(log_filename))[0]

    outdirname = os.path.join(outdirname, baseoutputname, net_string)
    os.makedirs(outdirname, exist_ok=True)

    make_plot(matches, conf_plot, mainfield_spec, target_col, outdirname, level=level)
    make_table(matches, conf_table, mainfield_spec, target_col, outdirname, level=level)



def make_plot(matches, conf_plot, mainfield_spec, target_col, outdirname, level=-3):
    pv = conf_plot['main_field_value']
    matches_plot = [m for m in matches if check_where(m.split('/')[level], mainfield_spec, pv)]
    ds_param_name, plot_field_spec = conf_plot['plot_field_spec']
    field_train = target_col + '_train'
    field_val = target_col + '_validation'
    field_test = target_col + '_test'
    all_fields = [ds_param_name, 'epoch', field_train, field_val, field_test]
    data = []
    for match in matches_plot:
        x = get_field(match.split('/')[level], plot_field_spec)
        epmax, y_val = read_max_value(match, field_val, 'epoch')
        y_train = read_value(match, 'epoch', epmax, field_train)
        y_test = read_value(match, 'epoch', epmax, field_test)
        data.append((tryfloat(x), epmax, y_train, y_val, y_test))

    sorted_data = sorted(data)
    df = pd.DataFrame(sorted_data, columns=all_fields)
    outpath_plot = os.path.join(outdirname, mainfield_spec[0] + pv + "_" + target_col + ".txt")
    df.to_csv(outpath_plot, index=False, float_format='%.6g', **pd_csv_kwargs)


def make_table(matches, conf_table, mainfield_spec, target_col, outdirname, level=-3):
    tv = conf_table['main_field_value']
    matches_table = [m for m in matches if check_where(m.split('/')[level], mainfield_spec, tv)]

    all_fields = [fn for fn, fs in conf_table['table_field_spec']]

    log_field_train = target_col + '_train'
    log_field_val = target_col + '_validation'
    log_field_test = target_col + '_test'
    all_fields += ['epoch', log_field_train, log_field_val, log_field_test]

    data = []
    for match in matches_table:
        # get values identifying this match as from specifications
        tables_field_values = [get_field(match.split('/')[level], tfspec, with_subfields=False) \
                                   for tfn, tfspec in conf_table['table_field_spec']]

        epmax, y_val = read_max_value(match, log_field_val, 'epoch')
        y_train = read_value(match, 'epoch', epmax, log_field_train)
        y_test = read_value(match, 'epoch', epmax, log_field_test)

        data.append(tuple(tables_field_values)+(epmax, y_train, y_val, y_test))

    sorted_data = sorted(data)
    df = pd.DataFrame(sorted_data, columns=all_fields)
    outpath_table = os.path.join(outdirname, mainfield_spec[0] + tv + "_" + target_col + ".txt")
    df.to_csv(outpath_table, index=False, float_format='%.6g', **pd_csv_kwargs)


def name_with_wildcard(ds_dir, field_spec):
    mainfield = field_spec[0]
    tag = '-' + mainfield + get_field(ds_dir, (mainfield,))
    pre_ds_dir, _, post_ds_dir = ds_dir.partition(tag)
    name_with_wildcard = pre_ds_dir + '-' + mainfield + 'W' + post_ds_dir
    return name_with_wildcard


def collect_data_across_ds(base_dir, ds_dirs, net_dirs, log_file, collect_dict, outdirname="."):
    for ds_dir, net_dir in product(ds_dirs, net_dirs):
        collect_data_across_ds_single(base_dir, ds_dir, net_dir, log_file, collect_dict, outdirname=outdirname)


def get_field(string, field_spec, with_subfields=True):
    l = len(field_spec)

    if l == 0 or l > 2:
        raise ValueError("Not implemented tuple length `{:}`, found field spec `{:}`".format(l, field_spec))

    m = re.split('(-|^)' + field_spec[0], string)
    try:
        after = m[2]
        ss1 = re.split('(-[a-zA-Z]|$)', after)[0]
    except:
        ss1 = ''

    if l == 1:
        if with_subfields:
            return ss1
        else:
            ss1val = re.split('(_|$)', ss1)[0]
            return ss1val

    m = re.search('(_|^)' + field_spec[1] + '([\.\-\,A-Za-z0-9]+)' + '(_|$)', ss1)

    if m is None:
        ss2 = ''
    else:
        ss2 = m.group(2)

    return ss2


def check_where(string, field_spec, value):
    return get_field(string, field_spec, with_subfields=False) == value


