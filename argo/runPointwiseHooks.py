import sys, os
sys.path.insert(0, os.getcwd())
from argo.core.utils.argo_utils import load_class, eval_file
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='Run pointwise hooks (those hooks not making average over several training steps but depending only on the current step) on saved argo models.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('conffile', nargs='+', help='The config files associated to the training of the model to load.')
parser.add_argument('--hooksconffile', '-hconf', help='The config file with the hooks to run.')
parser.add_argument('--global_step', nargs='+', type=int, help='The global_steps at which we want to restore the model. Default is the last one found in the folder.', default=[None])
parser.add_argument('--gpu', help='GPU where to run on.', default='0')
parser.add_argument('--seed', type=int, help='seed to randomize tables.', default=0)
parser.add_argument('--modelClassDir', '-md', help='directory where where we can find the file containing the class of the model to load (relative to where I launch this script, e.g. `core` or `prediction.core`, ...)', default="core")

args = parser.parse_args()

gpu = args.gpu
seed = args.seed
global_steps = args.global_step

conffiles = args.conffile
hooksconffile = args.hooksconffile
hooksconfig = eval_file(hooksconffile)
fix_period = 1
hook_keys = []

for hookkey, hookdict in hooksconfig.items():
    hookdict["period"] = fix_period
    hook_keys.append(hookkey)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

model_class_base_path = args.modelClassDir
model_class_base_path.replace("/", ".")

def load_and_run_hook(conf_file, global_step):
    import tensorflow as tf
    from datasets.Dataset import Dataset
    from argo.core.ArgoLauncher import ArgoLauncher

    tf.reset_default_graph()

    # ######################################################
    # # LOAD THE WHOLE MODEL WITH ITS OWN MONITOREDSESSION #
    # ######################################################
    #
    dataset_conf, model_parameters, config = ArgoLauncher.process_conf_file(conf_file)
    config.update(hooksconfig)

    #remove hooks that I do not want to trigger
    config["save_summaries"] = False
    config["save_model"] = False
    config["stats_period"] = 17e300 # an insanely large number, one of the biggest int before inf
    hooks_to_remove = ['LoggingMeanTensorsHook',
                       'GradientsHook',
                       ]
    for key in hooks_to_remove:
        config.pop(key, None)

    dataset = Dataset.load_dataset(dataset_conf)

    ArgoTFDeepLearningModelClass = load_class(model_parameters["model"], base_path=model_class_base_path)
    # add information about the dataset for the launchable construction, needed in view of future keras compatibility
    # try catch to allow compatibility for datasets which do not have labels (see Dataset interface)
    try:
        output_shape = dataset.y_shape
    except ValueError:
        output_shape = None

    dataset_info = {"output_shape" : output_shape,
                    "input_shape" : dataset.x_shape_train}

    model_parameters.update(dataset_info)

    model_dir = os.path.split(os.path.dirname(conf_file))[0]

    try:
        output_shape = dataset.y_shape
    except ValueError:
        output_shape = None

    dataset_info = {"output_shape" : output_shape,
                    "input_shape" : dataset.x_shape_train}

    model_parameters.update(dataset_info)

    model = ArgoTFDeepLearningModelClass(model_parameters, model_dir, gpu=gpu, seed=seed)
    model.init(dataset)

    # network = model._network
    # network.init_saver()

    x_shape = (1,) + model.x_shape['train']

    # # I want input shape but I don't want to pass by the handle, which might have more None shapes (if loop dataset has cropping)
    # train_loop_iter, _ = dataset.get_dataset_iterator(1, "train", shuffle=1, repeat=1, augment=1)
    # x_shape = train_loop_iter.get_next()[0].shape.as_list()
    # for i,d in enumerate(x_shape):
    #     if d is None:
    #         x_shape[i] = 1

    model._init_session_saver()
    model.create_session(model_parameters, config)
    #if global_step is None it will restore the last checkpoint in the folder model._checkpoint_dir, you can pass global_step to restore a particular chackpoint
    model.restore(global_step = global_step)
    # this is needed in case global_step was None, to load last step
    global_step = model.get_raw_session().run(model.global_step)

    # I force the trigger for the hooks in the config file
    max_steps = model._get_steps(fix_period, model._time_reference_str)

    # need extra list cos cannot remove elements while iterating
    to_remove = []
    for hook in model.hooks:
        if type(hook).__name__ in hook_keys:
            hook._timer.reset()
            hook._timer.update_last_triggered_step(global_step - max_steps)
        else:
            to_remove.append(hook)

    for h in to_remove:
        model.hooks.remove(h)

    # two times to trigger the hooks, since first step they are disabled by design
    gs = model.sess.run(model.global_step, feed_dict={model.raw_x: np.zeros(x_shape)})
    gs = model.sess.run(model.global_step, feed_dict={model.raw_x: np.zeros(x_shape)})



for conf_file in conffiles:
    for global_step in global_steps:
        load_and_run_hook(conf_file, global_step)
