import sys
import argparse
from core.argo.core.ArgoLauncher import ArgoLauncher
from core.AdversarialLauncher import AdversarialLauncher

parser = argparse.ArgumentParser(description='Run Adversarial Examples', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('confFile', help='The config file for the adversarial model')
parser.add_argument('parallelism', choices=['single', 'pool', 'distributed', 'stats'], help="The parallelism option")

args = parser.parse_args()
argv = ["dummy", args.confFile, args.parallelism]
dataset_conf, model_parameters, config, parallelism = ArgoLauncher.process_args(argv)


launcher = AdversarialLauncher()
launcher.run(dataset_conf, model_parameters, config, parallelism)
