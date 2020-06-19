import sys, os
sys.path.insert(0, os.getcwd())
import argparse
from argo.core.ArgoLauncher import ArgoLauncher
from argo.core.TrainingLauncher import TrainingLauncher

parser = argparse.ArgumentParser(description='Train a Model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('confFile', help='The config file for the training of the model')
parser.add_argument('parallelism', choices=['single', 'pool', 'distributed', 'stats'], help="The parallelism option")

args = parser.parse_args()
argv = ["dummy", args.confFile, args.parallelism]

dataset_params, model_params, config, parallelism = ArgoLauncher.process_args(argv)

launcher = TrainingLauncher()
launcher.run(dataset_params, model_params, config, parallelism)
