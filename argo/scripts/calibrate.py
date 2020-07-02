#!/usr/bin/env python
# coding: utf-8
import argparse

from datasets.Dataset import TRAIN_LOOP, TRAIN, VALIDATION, TEST, \
                    TRAIN_SHUFFLED, VALIDATION_SHUFFLED, TEST_SHUFFLED

from core.Calibrator import Calibrator
import os
from itertools import product
import pprint
# import tensorflow as tf
import copy
import time

from datetime import timedelta

def make_list(l):
    return l if isinstance(l, list) else [l]

def write_conf_file(outputname, conf):
    f = open(outputname, 'w')
    f.write(pprint.pformat(conf))
    f.close()

def write_final_log(outputname, startTime, endTime):
    f = open(outputname, 'w')
    elapsed = endTime - startTime
    f.write("time: "+str(timedelta(seconds=elapsed))+"\n")
    f.write("Done")

    f.close()

# def create_session(gpu=0, seed=0):
#     # CREATE SESSION
#     tf.set_random_seed(seed)
#     # tf.reset_default_graph()
#
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = '{:d}'.format(gpu)
#     sess_config = tf.ConfigProto()
#     sess_config.gpu_options.allow_growth = True
#     sess = tf.Session(config=sess_config)
#     return sess


implemented_methods = ['israelian', 'mcsampling', 'lastlayer', 'lastlayermu', 'finetune']
implemented_covcals = ['scalar', 'tril', 'fix']

parser = argparse.ArgumentParser(description='Calibrate an already trained regression Model.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('conffile', help='The config file for Calibration.')
# parser.add_argument('conffile', help='The config file associated to the training of the model to load.')
# parser.add_argument('--outputdir', help='Output directory.')
# parser.add_argument('--nepochs', type=int, help='Epochs for calibration.', default=100)
# parser.add_argument('--method', choices=implemented_methods, help='Method to choose.', default='israelian')
# parser.add_argument('--covcal', choices=implemented_covcals, help='How to calibrate the covariance', default='scalar')
# parser.add_argument('--global_step', help='The global_step at which we want to restore the model. Default is the last one found in the folder.', default=None)
# parser.add_argument('--gpu', type=int, help='GPU where to run on.', default=0)
# parser.add_argument('--seed', type=int, help='seed to randomize tables.', default=100)
# # parser.add_argument('--modelClassDir', '-md', help='directory where where we can find the file containing the class of the model to load (relative to where I launch this script, e.g. `core` or `prediction.core`, ...)', default="core")
# parser.add_argument('--use_alpha', type=bool, help='use alpha divergence', default=False)
# parser.add_argument('--alpha_parameter', type=float, help='value alpha divergence', default=0)
# parser.add_argument('--n_samples', type=int, help='samples from training', default=1)
args = parser.parse_args()

with open(args.conffile, 'r') as fstream:
    # load the three dictionaries
    config = eval(fstream.read())

conf_list = [copy.deepcopy(dict(zip(config.keys(), p))) for p in product(*map(make_list, config.values()))]

for conf in conf_list:

    conffile = conf['conffile']
    gpu = conf['gpu']
    seed = conf['seed']
    global_step = conf['global_step']

    method = conf['method']
    if method not in implemented_methods:
        raise Exception("method should be in: {:}".format(implemented_methods))

    covcal = conf['covcal']
    if covcal not in implemented_covcals:
        raise Exception("covcal should be in: {:}".format(implemented_covcals))

    output_dir = conf['outputdir']
    nepochs = conf['nepochs']
    alpha = conf.get('alpha', 0) #default is 0, so no use if not specified

    n_samples = conf['n_samples']

    flow_params = conf.get('flow', None)
    # flow_name, flow_kwargs = flow_tuple

    optimizer_tuple = conf['optimizer_tuple']

    ci_class_name, ci_kwargs = conf['confidence_intervals']
    # ci_class_name = conf['confidence_intervals']['class']
    ci_period = ci_kwargs.pop('period')
    ci_datasets = ci_kwargs.pop('datasets', [VALIDATION, TEST])


    calibrator = Calibrator(conffile, global_step, method, covcal, flow_params, optimizer_tuple, ci_class_name, alpha=alpha,
                            n_samples=n_samples, gpu=gpu, seed=seed, base_dir=output_dir)

    log_dir = calibrator._log_dir
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir + '/experiment.log'


    if not os.path.isfile(log_file):
        write_conf_file(log_dir + '/experiment.conf', conf)
        startTime = time.time()
        calibrator.prepare_for_calibration(dataset_eval_names=[TRAIN, VALIDATION, TEST],
                                           dataset_ci_names=ci_datasets, ci_period=ci_period,
                                           **ci_kwargs
                                           )
        calibrator.calibrate(nepochs) #, [TRAIN_SHUFFLED])
        endTime = time.time()
        write_final_log(log_file, startTime, endTime)

    else:
        print("found job done: {:}".format(log_file))

    calibrator.release()




