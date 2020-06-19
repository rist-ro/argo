import sys, os

sys.path.insert(0, os.getcwd())
from argo.core.utils.argo_utils import load_class, eval_file
import argparse
import os
import numpy as np
import re
from argo.core.argoLogging import get_logger
import traceback

tf_logging = get_logger()

SPLIT_FIRST_HALF = 'first_half'
SPLIT_SECOND_HALF = 'second_half'
SPLIT_NONE = 'None'
'''
--------------------------------------------
                HOW TO USE
--------------------------------------------

from vae directory run: 
    python runHooksConf.py examples/hooks/AudioHooks.conf

'''

parser = argparse.ArgumentParser(description='Run Hooks on VAE model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('hookConfFile',
                    help='The config file in which you specify what hooks you want to run and the models')

args = parser.parse_args()

hooksconfig = eval_file(args.hookConfFile)

model_names = hooksconfig['model_names']
model_names_ignore = hooksconfig['model_names_ignore']
run_all_global_steps = hooksconfig['run_all_global_steps']
global_steps = hooksconfig['global_steps']
seed = hooksconfig['seed']
gpu = str(hooksconfig['gpu'])
base_dir = hooksconfig['base_dir']
hooks = hooksconfig['hooks']
split = hooksconfig['split']

if not base_dir.endswith('/'):
    base_dir += '/'

# if you want to run experiments on all the models in base_dir than leave model_names empty
if len(model_names) == 0:
    model_names = [model_name for model_name in os.listdir(base_dir) if model_name not in model_names_ignore]
    model_names = sorted(model_names)

    if split != SPLIT_NONE:
        middle = len(model_names) // 2
        if split == SPLIT_FIRST_HALF:
            model_names = model_names[:middle]
        elif split == SPLIT_SECOND_HALF:
            model_names = model_names[middle:]

model_paths = [base_dir + model_name + '/' for model_name in model_names]
model_paths = list(filter(lambda x: not x.endswith('.png/'), model_paths))

tf_logging.info('Running hooks on {} experiments'.format(len(model_paths)))

# if you want to use the last global step for all the experiments leave the list empty
if len(global_steps) == 0 or run_all_global_steps:
    for model_path in model_paths:
        saved_models = os.listdir(model_path + 'saved_models')
        global_steps_set = set()

        for saved_model in saved_models:
            if saved_model.startswith('events') or saved_model == 'checkpoint':
                continue
            splitted = re.split(r'(\d+)', saved_model)
            global_step = int(splitted[1])
            global_steps_set.add(global_step)

        global_steps_set = sorted(global_steps_set)

        if run_all_global_steps or len(global_steps_set) == 0:
            global_steps.append(global_steps_set)
        else:
            global_steps.append(global_steps_set[-1])

# if you want to use the same global steps for all the models just specify it once
elif len(global_steps) < len(model_names):
    global_steps = [global_steps[0]] * len(model_names)

tf_logging.info('Global_steps:')
print(*global_steps, sep='\n')
print('\n')

conffiles_global_steps = [(model_path + 'experiment.conf', global_step) for model_path, global_step in
                          zip(model_paths, global_steps)]

fix_period = 1
hook_keys = []

for hookkey, hookdict in hooks.items():
    hookdict["period"] = fix_period
    hook_keys.append(hookkey)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

model_class_base_path = 'core'
model_class_base_path.replace("/", ".")


def load_and_run_hook(conf_file, global_steps_list):
    import tensorflow as tf
    from datasets.Dataset import Dataset
    from argo.core.ArgoLauncher import ArgoLauncher

    tf.reset_default_graph()

    # ######################################################
    # # LOAD THE WHOLE MODEL WITH ITS OWN MONITOREDSESSION #
    # ######################################################
    #
    dataset_conf, model_parameters, config = ArgoLauncher.process_conf_file(conf_file)
    if 'WavReconstructHook' not in hooks:
        config.pop('WavReconstructHook')
    if 'WavGenerateHook' not in hooks:
        config.pop('WavGenerateHook')

    config.update(hooks)

    # remove hooks that I do not want to trigger
    config["save_summaries"] = False
    config["save_model"] = False
    config["stats_period"] = 17e300  # an insanely large number, one of the biggest int before inf
    hooks_to_remove = [
        'LoggingMeanTensorsHook',
        'GradientsHook',
    ]
    for key in hooks_to_remove:
        config.pop(key, None)

    dataset = Dataset.load_dataset(dataset_conf)
    ArgoTFDeepLearningModelClass = load_class(model_parameters["model"], base_path=model_class_base_path)

    model_dir = os.path.split(os.path.dirname(conf_file))[0]
    model = ArgoTFDeepLearningModelClass(model_parameters, model_dir, gpu=gpu, seed=seed)
    model.init(dataset)

    # network = model._network
    # network.init_saver()

    x_shape = (1,) + tuple(model.x_shape['train'])

    model._init_session_saver()
    model.create_session(model_parameters, config)

    # if global_step is None it will restore the last checkpoint in the folder model._checkpoint_dir, you can pass global_step to restore a particular chackpoint
    for global_step in global_steps_list:
        tf_logging.info('...Running global step... ' + str(global_step))
        try:
            model.restore(global_step=global_step)
        except Exception:
            print('-----LOAD EXCEPTION: could not LOAD model at step', global_step)
            continue
        # this is needed in case global_step was None, to load last step
        global_step = model.get_raw_session().run(model.global_step)

        # I force the trigger for the hooks in the config file
        max_steps = model._get_steps(fix_period, model._time_reference_str)

        # need extra list cos cannot remove elements while iterating
        to_remove = []
        for hook in model.hooks:
            if type(hook).__name__ in hook_keys:
                hook._timer.reset()
                hook.before_training(model.sess)
                hook._timer.update_last_triggered_step(global_step - max_steps)
            else:
                to_remove.append(hook)

        for h in to_remove:
            model.hooks.remove(h)

        # two times to trigger the hooks, since first step they are disabled by design
        gs = model.sess.run(model.global_step, feed_dict={model.raw_x: np.zeros(x_shape)})
        gs = model.sess.run(model.global_step, feed_dict={model.raw_x: np.zeros(x_shape)})

    tf_logging.info('Finished with model...')


for conf_file, global_steps_set in conffiles_global_steps:
    if not isinstance(global_steps_set, list):
        global_steps_set = [global_steps_set]

    # for global_step in global_steps_set:
    #     if global_step != -1:
    try:
        load_and_run_hook(conf_file, global_steps_set)
    except Exception as excp:
        print('-----UNEXPECTED EXCEPTION... moving forward. \nModel: {} \n{}'.format(conf_file, excp))
        traceback.print_exc()
