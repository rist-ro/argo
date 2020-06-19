import argparse

from argo.core.utils.PlotCurves import PlotCurves

parser = argparse.ArgumentParser(description='Plot curves from different txt log files to compare experiments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('conf_file', help='The config file with the details of the experiments')

args = parser.parse_args()
argv = ["dummy", args.conf_file]

conf_file = eval(open(args.conf_file, 'r').read())


for stp in [0.1, 0.2, 0.3, 0.4]:
    for zsc in [1.0, 2.0, 4.0]:
        conf = conf_file.copy()

        print(stp, zsc)
        conf['where'] = [('stp', str(stp), 1)]
        conf['block'] = [("zsc", "1.0", 4), ("zsc", "1.5", 4), ("zsc", "2.0", 4), ("zsc", "2.5", 4), ("zsc", "3.0", 4), ("zsc", "3.5", 4), ("zsc", "4.0", 4), ("zsc", "6.0", 4), ("zsc", "8.0", 4)]
        conf['block'].remove(("zsc", str(zsc), 4))
        conf['legend'] = [("rec", 1, "model "), ("d", 1, "denoising ")]

        tag = 'various-models' + '_stp' + str(stp) + '_zsc' + str(zsc)
        conf['tag'] = tag

        pc = PlotCurves()
        pc.plot(**conf)
