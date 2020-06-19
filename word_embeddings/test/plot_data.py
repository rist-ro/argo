import json
import os
import matplotlib
matplotlib.use('Agg')
from core.plotting import initialize_plot, finalize_plot, load_data_dict, plot_data, merge_data_dicts

# matplotlib.rc('text', usetex=True)
# matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Lucida']})
matplotlib.rcParams.update({'savefig.dpi': '300'})

fontsize=20
fonttitlesize=fontsize
fontaxeslabelsize=16
fontlegendsize=16

matplotlib.rcParams.update({'font.size': fontsize})
# matplotlib.rcParams.update({'font.weight': 'bold'})
# matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

matplotlib.rcParams.update({'legend.frameon': False})
matplotlib.rcParams.update({'legend.fontsize': fontlegendsize})
matplotlib.rcParams.update({'legend.borderaxespad': 1.0})

matplotlib.rcParams.update({'lines.linewidth': 4.0})
matplotlib.rcParams.update({'lines.markersize': 9})

matplotlib.rcParams.update({'axes.titlesize': fonttitlesize})
matplotlib.rcParams.update({'axes.labelsize': fontaxeslabelsize})

# matplotlib.rcParams.update({'axes.labelpad': 16.0})
# matplotlib.rcParams.update({'xtick.major.pad': 10.0})
# matplotlib.rcParams.update({'ytick.major.pad': 5.0})

figsize = matplotlib.rcParams['figure.figsize']
figsize[0] = 6.4 * 3
figsize[1] = 4.8 * 3



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Plot from data files (extract_stats.py should be run first).',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('inputfile', nargs='+', help='The data files from which to plot.\
                         Each data file should be a dictionary containing: \n\
                         {plot_method: .., x-label: .., y-label: .., data: .. }, where data is actually a list of dictionaries\
                         \ndata = [{xs: .., ys: .., label: ..}, {xs: .., ys: .., label: ..}, ... ]')
    parser.add_argument('--legendloc', type=str, default=None, help='location for the legend if the standard is not ok')
    parser.add_argument('--outputname', '-o', help='how to save output figure. If it is not present,\
                        only one inputfile is expected. (output will be input.png)')
    parser.add_argument('--title', type=str, default='', help='Title of the plot.')
    parser.add_argument('--xlabel', type=str, default='', help='The x-label, if want to overwrite the default one in the inputfile.')
    parser.add_argument('--ylabel', type=str, default='', help='The y-label, if want to overwrite the default one in the inputfile.')
    parser.add_argument('--xlim', nargs=2, type=float, help='The x-lims to use. (xlow, xhigh)')
    parser.add_argument('--ylim', nargs=2, type=float, help='The y-lims to use. (ylow, yhigh)')
    parser.add_argument('--plot_method', type=str, default=None, help='Plot method to use if you want to overwrite the one in the data.')
    parser.add_argument('--plot_args', nargs='+', type=str, default=[], help='extra args to pass to the plot_method.')
    parser.add_argument('--plot_kwargs', type=json.loads, default={}, help='extra kwargs to pass to the plot_method. \
                            json reader expects \'{\"key1\":\"value1\", \"key2\":\"value2\", .. }\'')

    # parser.add_argument('--labels', type=str, nargs='+', default=None, help='Overwrite the labels for the input files.')
    args = parser.parse_args()
    title = args.title
    xlim=args.xlim
    ylim=args.ylim
    legendloc=args.legendloc
    outname=args.outputname

    plot_args = args.plot_args

    plot_kwargs = {'alpha':0.5}
    plot_kwargs.update(args.plot_kwargs)

    if not outname:
        if len(args.inputfile)>1:
            raise ValueError("Outputname is not present. Only one inputfile is expected.\
                            (output will be input.png). To plot from multiple input file\
                            you need to set outputname.")
        else:
            outname = os.path.splitext(args.inputfile[0])[0]+".png"

    data_dict_list = [load_data_dict(dataname) for dataname in args.inputfile]
    data_dict = merge_data_dicts(data_dict_list)

    plot_method = args.plot_method
    xlabel=args.xlabel
    ylabel=args.ylabel

    if not plot_method:
        plot_method = data_dict['plot_method']

    if not xlabel:
        xlabel = data_dict['x-label']

    if not ylabel:
        ylabel = data_dict['y-label']

    axes = initialize_plot(title, xlabel, ylabel)

    plot_data(data_dict['data'], axes, plot_method, plot_args, plot_kwargs)
    finalize_plot(axes, outname, xlim=xlim, ylim=ylim, legendloc=legendloc)

    # def hist(data, outname,  binsnumber, title= '', xlabel='', legendloc=None):
    #     axes=initialize_plot()
    #     # plot_data(data, axes, axes.hist, plotkwargs={'alpha':0.5}, title="Norm of the Vectors of the embeddings", xlim=(dmin, dmax), xlabel="norms")
    #     xs_list = [dc['xs'] for dc in data]
    #     dmin = min(map(min,xs_list))
    #     dmax = max(map(max,xs_list))
    #     delta = dmax - dmin
    #     dmin = dmin - 0.05 * delta
    #     dmax = dmax + 0.05 * delta
    #
    #     bins = np.linspace(dmin, dmax, binsnumber)
    #     for dc in data:
    #         dc['ys']=bins
    #
    #     plot_data(data, axes, axes.hist, plotkwargs={'alpha':0.5}, title=title, xlabel=xlabel, legendloc=legendloc)
    #     finalize_plot(axes, outname)
    #
    # def plot(data, outname, title= '', xlabel='', ylabel='', legendloc=None):
    #     axes=initialize_plot()
    #     plot_data(data, axes, axes.plot, title=title, plotkwargs={'alpha':0.5}, xlabel=xlabel, ylabel=ylabel)
    #     finalize_plot(axes, outname)
    # if args.command=='plot':
    #     plot(data, outname, legendloc=legendloc)
    # elif args.command=='histogram':
    #     hist(data, outname, args.num, legendloc=legendloc)
