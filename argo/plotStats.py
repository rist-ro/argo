import matplotlib.pyplot as plt
import os
import numpy as np
import random


def index_of(str_value, list_):
    for i, value in enumerate(list_):
        if str_value in value:
            return i
    return None


def interesting_features_from_name(model_name, features=['wu', 'hl', 'oc', 'sc', 'lc', 'lp', 'vl', 'hc']):
    model_features = model_name.split('-')
    final_model_features = []
    for model_feature in model_features:
        for feature in features:
            if feature in model_feature:
                final_model_features.append(model_feature)

    return '__'.join(final_model_features)


bits_dim_valid = lambda x: True

target_files_and_index = [
    ('cost_function_KL.txt', 'KL_validation', lambda x: x < 30),
    ('cost_function_RL.txt', 'RL_validation', lambda x: x > -16000),
    ('cost_function_RL.txt', 'RL_train_loop', lambda x: x > -10000),
    ('reconstr_metrics_x_validation.txt', 'MCD', lambda x: x < 30),
    ('reconstr_metrics_x_validation.txt', 'PSNR', lambda x: x > 16),
    ('reconstr_metrics_x_train.txt', 'MCD', lambda x: x < 30),
    ('reconstr_metrics_x_train.txt', 'PSNR', lambda x: x > 16),
    ('bits_dim.txt', 'b/d_train_loop', bits_dim_valid),
    ('bits_dim.txt', 'b/d_validation',  bits_dim_valid)
]

base_dir = '/data2/rcolt/HB-VAE-changed_prior/DigiScope/'

model_names = [base_dir + dir_path for dir_path in os.listdir(base_dir) if not dir_path.endswith('.png')]
# model_names = list(filter(lambda x: 'lc4' in x, model_names))

cmap = plt.get_cmap('tab20c').colors[:-4] + plt.get_cmap('tab20b').colors[-12:]
colors = cmap
# colors = [cmap(i) for i in np.linspace(0, 1, len(model_names))]
# random.shuffle(colors)

for file_name, metric_name, validate_metric in target_files_and_index:
    color_idx = 0
    x_list = []
    y_list = []
    name_list = []

    for i, model_name in enumerate(model_names):
        interest_model_name = str(i + 1) + '. ' + interesting_features_from_name(model_name)#model_name.split('-')[3].split('_')[-1] + '_' + '-'.join(model_name.split('-')[-10:])

        full_file_name = model_name + '/' + file_name
        if os.path.isfile(full_file_name):
            with open(full_file_name, 'r+') as input_file:
                lines = input_file.readlines()
                x = []
                y = []

                metric_index = index_of(metric_name, lines[0].split('\t'))

                for line in lines[1:]:
                    split_line = line.strip().split('\t')
                    x.append(int(split_line[0]))
                    y.append(float(split_line[metric_index]))

                name_list.append(interest_model_name)
                x_list.append(x)
                y_list.append(y)
        else:
            print('Cannot find the file:', full_file_name, model_name)

    for x_, y_, name in zip(x_list, y_list, name_list):
        if len(y_) > 0 and validate_metric(y_[-1]):
            plt.plot(x_, y_, label=name + ' : ' + str(y_[-1]), color=colors[color_idx])
            color_idx += 1

    plt.legend(loc=2, prop={'size': 6}, bbox_to_anchor=(1, 1))
    plt.tight_layout()
    ds_key_name = file_name.split('.')[0].split('_')[-1]
    plt.savefig('{}/{}_{}.png'.format(base_dir, ds_key_name, metric_name.replace('/', '_')))
    plt.clf()
