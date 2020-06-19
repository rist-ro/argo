import sys
import argparse

from prediction.argo.core.utils.collect_across import collect_data_across_ds_before

parser = argparse.ArgumentParser(description='Collect curves for a single network across datasets', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('conf_file', help='The config file with the details of the data to extract')

args = parser.parse_args()

conf_file = eval(open(args.conf_file, 'r').read())

collect_data_across_ds_before(**conf_file)
