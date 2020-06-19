import sys
import argparse

from prediction.argo.core.utils.collect_table import collect_table_across_ds

parser = argparse.ArgumentParser(description='Collect table for a single network across datasets', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('conf_file', help='The config file with the details of the data to extract')

args = parser.parse_args()

conf_file = eval(open(args.conf_file, 'r').read())

collect_table_across_ds(**conf_file)
