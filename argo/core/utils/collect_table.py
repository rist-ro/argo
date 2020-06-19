import os, re
from glob import glob
from pprint import pprint
import pandas as pd
import numpy as np
from itertools import product
from .collect_across import check_matches, get_ds_field_value, pd_csv_kwargs


#
# collect_dict = {
#     'main_field_spec' : ('a',),
#
#     'plot' : {
#         'main_field_value' : 'float',
#         'plot_field_spec' : ('alpha', ('a','v')),
#     },
#
#     'table' : {
#         'main_field_value' : 'limit',
#         'table_field_spec' : [
#             ('ntop', ('a', 't')),
#             ('weighted', ('a', 'w')),
#             ('fraction', ('a', 'f')),
#         ],
#     }
# }

def read_row_max(path, field, columns):
    df = pd.read_csv(path, **pd_csv_kwargs)
    imax = df[field].idxmax()
    return list(np.array(df[columns].loc[imax]))

def _isclose(x, val):
    if isinstance(x, float) and isinstance(val, float):
        return np.isclose(x, val, atol=1e-8)
    else:
        return x == val

def read_ref_row(path, field, value, columns):
    df = pd.read_csv(path, **pd_csv_kwargs)
    ref_df = df.loc[np.array([_isclose(x, value) for x in df[field]], dtype=bool)]

    if ref_df.empty:
        raise ValueError("Could not find reference value `{:}={:}` in file `{:}`".format(field, value, path))

    #if not empty
    if ref_df.shape[0]>1:
        raise Exception('found more than one reference {:}'.format(df))
    return list(np.array(ref_df[columns].iloc[0].array))

 # tables = {
 #     'limit' : {
 #         'log_file': "alimit_auc.txt",
 #         'target_col' : 'auc',
 #         'extra_columns' : ['ntop', 'weighted', 'fraction'],
 #     },
 #
 #     'float' : {
 #         'log_file': "afloat_auc.txt",
 #         'target_col' : 'auc',
 #         'extra_columns' : ['alpha'],
 #         'reference' : ('alpha', 1.0),
 #     }
 # }

def collect_table_across_ds(base_dir, ds_dirs, net_dirs, tables, outdirname="."):
    os.makedirs(outdirname, exist_ok=True)
    ds_idxs = np.array([2,5,6,7])

    for table_name in tables:
        conf = tables[table_name]
        log_file = conf['log_file']
        target_col = conf['target_col']
        extra_columns = conf['extra_columns']
        ref_field, ref_value = conf.get('reference', (None, None))

        data = []
        col_name_train = target_col + '_train'
        col_name_val = target_col + '_validation'
        col_name_test = target_col + '_test'

        columns_to_return = [col_name_train, col_name_val, col_name_test] + extra_columns

        for ds_dir, net_dir in product(ds_dirs, net_dirs):

            dir_list = [base_dir, ds_dir, net_dir, log_file]
            matches = glob(os.path.join(*dir_list))
            if len(matches) == 0:
                raise ValueError("No file found matching the provided regexpr: `{:}`".format(os.path.join(*dir_list)))


            for match in matches:
                ds_level = -3
                ds_name = match.split('/')[ds_level]
                ds_name = '-'.join(np.array(ds_name.split('-'))[ds_idxs])
                data_row = read_row_max(match, col_name_val, columns_to_return)
                data.append([ds_name]+data_row)

                if ref_field is not None:
                    ref_row = read_ref_row(match, ref_field, ref_value, columns_to_return)
                    if len(ref_row)>0:
                        data.append([ds_name]+ref_row)

        sorted_data = sorted(data)

        all_fields = ['ds'] + columns_to_return
        df = pd.DataFrame(sorted_data, columns=all_fields)
        outputname = os.path.join(outdirname, table_name+"_"+target_col+".txt")
        df.to_csv(outputname, index=False, float_format='%.6g', **pd_csv_kwargs)




