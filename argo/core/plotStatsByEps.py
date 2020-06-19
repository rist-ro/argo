import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
import re
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Plot curves from different log files to compare experiments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('conf_file', help='The config file with the details of the experiments')

args = parser.parse_args()
argv = ["dummy", args.conf_file]

conf_file = eval(open(args.conf_file, 'r').read())

ALPHA = 0.6

base_plot_dir = conf_file['dirname']
base_csv_dir = conf_file['logs_directory']
attack_dir = "adv" + conf_file['attack'] + "_n100*"
csv_dirs = conf_file['csv_dirs']
ds_dir = conf_file['dataset']

stp_val = conf_file['stp']
quantities = conf_file['quantities']

fontsize = conf_file['font_size']
fonttitlesize = conf_file['font_title_size']
fontaxeslabelsize= conf_file['font_axes_label_size']
fontlegendsize = conf_file['font_legend_size']

figsize = matplotlib.rcParams['figure.figsize']
figsize[0] = conf_file['fig_size'][0]
figsize[1] = conf_file['fig_size'][1]

matplotlib.rcParams.update({'savefig.dpi': '300'})
matplotlib.rcParams.update({'font.size': fontsize})
matplotlib.rcParams.update({'legend.frameon': False})
matplotlib.rcParams.update({'legend.fontsize': fontlegendsize})
matplotlib.rcParams.update({'legend.borderaxespad': 1.0})
matplotlib.rcParams.update({'lines.linewidth': 4.0})
matplotlib.rcParams.update({'lines.markersize': 9})
matplotlib.rcParams.update({'lines.marker': 'o'})
matplotlib.rcParams.update({'errorbar.capsize': 7})
matplotlib.rcParams.update({'axes.titlesize': fonttitlesize})
matplotlib.rcParams.update({'axes.labelsize': fontaxeslabelsize})

nice_yaxis_names = {
    'nratio' : '$|\mu^*|/|\mu|$',
    'costh' : r'$\cos \, \theta$',
    'costh_mu' : r'$\cos \, \theta$',
    'costh_mu_adv' : r'$\cos \, \theta$',
    'adiff' : '$|\mu^*| - |\mu|$',
    'ndiff' : '$|\mu^* - \mu|$',
    'norm' : '$|\mu|$',
    'norm_adv' : '$|\mu^*|$',
    'cdist' : 'centroid dist',
    'cdist_adv' : 'centroid dist',
    'diameter_mu' : 'diameter',
    'diameter_mu_adv' : 'diameter',
}

def get_field(string, field_spec):
    l = len(field_spec)
    if l==0 or l>2:
        raise ValueError("Not implemented tuple length `{:}`, found field spec `{:}`".format(l, field_spec))
    m = re.search('(-|^)' + field_spec[0] + '([\._A-Za-z0-9\,]+)' + '(-|$)', string)
    if m is None:
        ss1 = '0'
    else:
        ss1 = m.group(2)
    if l==1:
        return ss1
    m = re.search('(_|^)' + field_spec[1] + '([\.A-Za-z0-9\,]+)' + '(_|$)', ss1)
    if m is None:
        ss2 = '0'
    else:
        ss2 = m.group(2)
    return ss2

def plot_grouped_curves(df, x, y, groupfield, pre_label):
    garr = [(label, df) for label, df in df.groupby(groupfield)]
    for groupval, df in garr:
        plabel = pre_label+'-'+groupfield+"{:}".format(groupval)
        xdata = df[x]
        pkwargs = {
            'alpha' : ALPHA,
            'label' : plabel
        }
        if y+'_avg' in df.columns and y+'_std' in df.columns:  
            ydata = df[y+'_avg']
            yerr = df[y+'_std']
            plt.errorbar(xdata, ydata, yerr=yerr, **pkwargs)
        else:
            ydata = df[y]
            plt.plot(xdata, ydata, **pkwargs)
        plt.legend()

def make_ds_plot(ds_dir, csv_dirs_dict, y, stp, plot_dir):
    x = 'eps'
    nice_ds_name = ds_dir.split("-")[0]
    for net_dir, transf_dir in csv_dirs_dict:
        stats_csv = 'statistics/stats-test.csv'
        dir_list = [base_csv_dir, ds_dir, net_dir, transf_dir, attack_dir, stats_csv]
        matches = list(set(glob(os.path.join(*dir_list))))
        groupfield = 'scale'
        plt.title(nice_ds_name+' stp={:}'.format(stp))
        plt.xlabel(x)
        plt.ylabel(nice_yaxis_names[y])
        df = pd.DataFrame({'ph': []})
        recv = ''
        if len(matches) > 0:
            for match in sorted(matches):
                recv = get_field(match.split('/')[-4], ('rec',))
                df = pd.read_csv(match, sep=' ')
                plot_grouped_curves(df, x, y, groupfield, recv)
    
    fields = [nice_ds_name, x, y, 'eps{:.2f}'.format(stp_val)]
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '-'.join(fields)+'.png'))
    plt.close()

os.makedirs(base_plot_dir, exist_ok=True)
for y in quantities:
    make_ds_plot(ds_dir, csv_dirs, y, stp_val, base_plot_dir)
