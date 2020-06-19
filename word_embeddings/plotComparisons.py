import sys
import argparse

from core.argo.core.utils.PlotCurves import plot_curves

parser = argparse.ArgumentParser(description='Plot curves from different txt flog iles to compare experiments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('conf_file', help='The config file with the details of the experiments')

args = parser.parse_args()
argv = ["dummy", args.conf_file]


conf_file = eval(open(args.conf_file, 'r').read())
    
plot_curves(**conf_file)
