import argparse
import os, sys
sys.path.insert(0, os.getcwd())

from core.utils.PlotImages import PlotImages

parser = argparse.ArgumentParser(description='Plot curves from different txt log files to compare experiments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('conf_file', help='The config file with the details of the experiments')

args = parser.parse_args()
argv = ["dummy", args.conf_file]

conf_file = eval(open(args.conf_file, 'r').read())

pc = PlotImages()
pc.plot(**conf_file)
