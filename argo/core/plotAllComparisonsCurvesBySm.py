import argparse

from argo.core.utils.PlotCurves import PlotCurves

parser = argparse.ArgumentParser(description='Plot curves from different txt log files to compare experiments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('conf_file', help='The config file with the details of the experiments')

args = parser.parse_args()
argv = ["dummy", args.conf_file]

conf_file = eval(open(args.conf_file, 'r').read())

# Turns floats of the form a.b into "a-b", so that the plots can be put on Overleaf
def latex_friendly(x):
    char_x = str(x)
    new_char_x = char_x[0] + '-' + char_x[2]
    return new_char_x


model = 'VAE'; stp = 0.1
conf = conf_file.copy()

conf['where'] = [("rec", model, 1), ("stp", str(stp), 1), ("d", "0", 1)]
conf['block'] = [("zsc", "1.5", 4), ("zsc", "2.0", 4), ("zsc", "2.5", 4), ("zsc", "3.0", 4), ("zsc", "3.5", 4), ("zsc", "4.0", 4), ("zsc", "6.0", 4), ("zsc", "8.0", 4), ("sm", "0.0", 1)]
conf['legend'] = [("ne", "sm", 1, "s ")]

tag = 'various-sm_' + 'C' + model + '_d0_stp' + latex_friendly(stp)
conf['tag'] = tag

pc = PlotCurves()
pc.plot(**conf)
