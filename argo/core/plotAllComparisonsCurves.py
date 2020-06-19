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
    new_char_x = char_x[0] + '-' + char_x[2:]
    return new_char_x


# Plot by stp, fixing the model type and factor
for model in ['AE', 'VAE']:
    if model == 'AE':
        for denoising in [0, 1]:
            conf = conf_file.copy()

            conf['where'] = [('rec', model, 1), ('d', str(denoising), 1)]
            conf['legend'] = [("stp", 1, "stp ")]

            tag = 'various-stp_' + model + '_d' + str(denoising)
            conf['tag'] = tag

            pc = PlotCurves()
            pc.plot(**conf)

    elif model == 'VAE':
        for denoising in [0, 1]:
            for zsc in [1.0, 2.0, 4.0]:
                conf = conf_file.copy()

                conf['where'] = [('rec', model, 1), ('d', str(denoising), 1), ('zsc', str(zsc), 4)]
                conf['legend'] = [("stp", 1, "stp ")]

                tag = 'various-stp_' + model + '_d' + str(denoising) + '_zsc' + latex_friendly(zsc)
                conf['tag'] = tag

                pc = PlotCurves()
                pc.plot(**conf)


# Plot by model, fixing stp and factor
for stp in [0.1, 0.2, 0.3, 0.4]:
    for zsc in [1.0, 2.0, 4.0]:
        conf = conf_file.copy()

        print(stp, zsc)
        conf['where'] = [('stp', str(stp), 1)]
        conf['block'] = [("zsc", "1.0", 4), ("zsc", "1.5", 4), ("zsc", "2.0", 4), ("zsc", "2.5", 4), ("zsc", "3.0", 4), ("zsc", "3.5", 4), ("zsc", "4.0", 4), ("zsc", "6.0", 4), ("zsc", "8.0", 4)]
        conf['block'].remove(("zsc", str(zsc), 4))
        conf['legend'] = [("rec", 1, "model "), ("d", 1, "denoising ")]

        tag = 'various-models' + '_stp' + latex_friendly(stp) + '_zsc' + latex_friendly(zsc)
        conf['tag'] = tag

        pc = PlotCurves()
        pc.plot(**conf)


# Plot by factor, fixing the model type (VAE or DVAE) nad stp
for stp in [0.1, 0.2, 0.3, 0.4]:
    for denoising in [0, 1]:
        conf = conf_file.copy()

        conf['where'] = [("rec", "VAE", 1), ("d", str(denoising), 1), ("stp", str(stp), 1)]
        conf['block'] = [("zsc", "6.0", 4), ("zsc", "8.0", 4)]
        conf['legend'] = [("zsc", 4, "factor ")]

        tag = 'various-zsc_' + 'VAE_d' + str(denoising) + '_stp' + latex_friendly(stp)
        conf['tag'] = tag

        pc = PlotCurves()
        pc.plot(**conf)
