import tensorflow as tf

from ..utils.argo_utils import eval_method_from_tuple
from .LearningRates import process_learning_rate, get_lr_id

import sys
import importlib
import pdb


def get_ilr_id(ilr):
    if isinstance(ilr, list):
        return "_".join(map(str, ilr))
    else:
        return str(ilr)


class TFOptimizers():

    @staticmethod
    def instantiate_optimizer(model, optimizer_tuple):

        optimizer_name = optimizer_tuple[0]
        optimizer_kwargs = optimizer_tuple[1]

        lr = process_learning_rate(optimizer_kwargs["learning_rate"], model.global_step, model.n_batches_per_epoch)

        # I want to copy because I want to modify it and I don't want to accidentally modify all the references around
        # in python references to a particular entry of a dictionary can be passed around and I might overwrite different task_opts
        optimizer_kwargs = optimizer_kwargs.copy()
        optimizer_kwargs.update({"learning_rate" : lr})

        try:
            # try to get the module from tf.train
            training_optimizer = eval_method_from_tuple(tf.train, (optimizer_name, optimizer_kwargs))
        except AttributeError as e:

            optimizer_kwargs["model"] = model
            try:
                # first try to load from argo.core.optimizers
                optimizer_module = importlib.import_module("." + optimizer_name, '.'.join(__name__.split('.')[:-1]))

                training_optimizer = eval_method_from_tuple(optimizer_module, (optimizer_name, optimizer_kwargs))

            except ImportError:
                try:
                    # second try to load from core.optimizers
                    optimizer_module = importlib.import_module("core.optimizers." + optimizer_name, '.'.join(__name__.split('.')[:-1]))

                    training_optimizer = eval_method_from_tuple(optimizer_module, (optimizer_name, optimizer_kwargs))

                except ImportError:
                    try:
                        # third try to load from core
                        optimizer_module = importlib.import_module("core." + optimizer_name, '.'.join(__name__.split('.')[:-1]))

                        training_optimizer = eval_method_from_tuple(optimizer_module, (optimizer_name, optimizer_kwargs))

                    
                    except ImportError:
                        try:
                            pdb.set_trace()
                            # next try to load kfac
                            import kfac

                            layer_collection = kfac.LayerCollection()
                            layer_collection.register_categorical_predictive_distribution(model.logits, name="logits")
                            # Register parameters. K-FAC needs to know about the inputs, outputs, and
                            # parameters of each conv/fully connected layer and the logits powering the
                            # posterior probability over classes.

                            tf.logging.info("Building LayerCollection.")
                            layer_collection.auto_register_layers()

                            # training_module = importlib.import_module("." + training_algorithm_name, '.'.join(__name__.split('.')[:-1]))
                            training_module = kfac
                            
                            kfac_kwargs = {
                                **optimizer_kwargs,
                                "layer_collection":   layer_collection,
                                "placement_strategy": "round_robin",
                                "cov_devices":        ["/gpu:0"],
                                "inv_devices":        ["/gpu:0"],
                            }
                            training_optimizer = eval_method_from_tuple(training_module, (optimizer_name, kfac_kwargs))

                        except Exception as e:
                            raise Exception("problem loading training algorithm: %s, kwargs: %s, exception: %s" % (
                                                            training_module, optimizer_kwargs, e)) from e

        return training_optimizer, lr

    @staticmethod
    def _get_kfac_parameters(optimizer_kwargs):
        _id = '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
        _id += '_ced' + str(optimizer_kwargs["cov_ema_decay"])
        _id += '_d' + str(optimizer_kwargs["damping"])
        _id += '_m' + str(optimizer_kwargs["momentum"])

        return _id

    @staticmethod
    def create_id(optimizer_tuple):
        # TODO huge 'if cascade'... each optimizer should have its own id.. or maybe in utils?
        # REPLY: I agree with this, the problem is that we have here also TF optimizer
        # for which we don't have a file where it is implemented. I don't like to have everything
        # in utils, since things should be defined where they are used. Utils is difficult to read
        # (Luigi 2019-05-26)
        optimizer_name, optimizer_kwargs = optimizer_tuple

        _id = ''


        if optimizer_name == 'GradientDescentOptimizer':
            _id += 'GD'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])


        elif optimizer_name == 'AdagradOptimizer':
            _id += 'AG'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])


        elif optimizer_name == 'RMSPropOptimizer':
            _id += 'RMS'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
            _id += '_m' + str(optimizer_kwargs["momentum"])
            _id += '_d' + str(optimizer_kwargs["decay"])
            _id += '_e' + str(optimizer_kwargs["epsilon"])


        elif optimizer_name == 'MomentumOptimizer':
            _id += 'A'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
            _id += '_m' + str(optimizer_kwargs["momentum"])
            _id += '_lk' + ('1' if str(optimizer_kwargs["use_locking"]) == 'True' else '0')
            _id += '_n' + ('1' if str(optimizer_kwargs["use_nesterov"]) == 'True' else '0')


        elif optimizer_name == 'ProximalGradientDescentOptimizer':
            _id += 'PGD'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
            _id += '_l1' + str(optimizer_kwargs["l1_regularization_strength"])
            _id += '_l2' + str(optimizer_kwargs["l2_regularization_strength"])
            _id += '_lk' + ('1' if str(optimizer_kwargs["use_locking"]) == 'True' else '0')


        elif optimizer_name == 'AdamOptimizer':
            _id += 'A'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
            _id += '_bo' + str(optimizer_kwargs["beta1"])
            _id += '_bt' + str(optimizer_kwargs["beta2"])

        
        #elif optimizer_name == 'MomentumOptimizer':
        #    _id += 'M'
        #    _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"]) 
        #    _id += '_m' + get_lr_id(optimizer_kwargs["momentum"])
        #    _id += '_l' + ("1" if optimizer_kwargs["use_locking"] else "0")
        #    _id += '_n' + ("1" if optimizer_kwargs["use_nesterov"] else "0")

        #elif optimizer_name == 'NesterovConstOptimizer':
        #    _id += 'NC'
        #    _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"]) 
        #    _id += '_m' + get_lr_id(optimizer_kwargs["momentum"])
        #    _id += '_l' + ("1" if optimizer_kwargs["use_locking"] else "0")
            
        elif optimizer_name == 'AdadeltaOptimizer':
            _id += 'Ad'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
            _id += '_r' + str(optimizer_kwargs["rho"])
            _id += '_e' + str(optimizer_kwargs["epsilon"])

        elif optimizer_name == 'NewtonMethodOptimizer':
            _id += 'N'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
            _id += '_d' + str(optimizer_kwargs["damping"])


        elif optimizer_name == 'SaddleFreeNewtonOptimizer':
            _id += 'SFN'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
            _id += '_d' + str(optimizer_kwargs["damping_values"])


        elif optimizer_name == 'NesterovConst':
            _id += 'NC'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
            _id += '_m' + str(optimizer_kwargs["momentum"])
            _id += '_lk' + ('1' if str(optimizer_kwargs["use_locking"]) == 'True' else '0')


        elif optimizer_name == 'NesterovNonconst':
            _id += 'NNC'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
            _id += '_a' + str(optimizer_kwargs["alpha"])
            _id += '_lk' + ('1' if str(optimizer_kwargs["use_locking"]) == 'True' else '0')


        elif optimizer_name == 'Indian':
            _id += 'IND'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
            _id += '_a' + str(optimizer_kwargs["alpha"])
            _id += '_b' + str(optimizer_kwargs["beta"])
            _id += '_g0' + str(optimizer_kwargs["gamma_0"])
            _id += '_gp' + str(optimizer_kwargs["gamma_power"])
            _id += '_iv' + str(optimizer_kwargs["init_velocity"])
            _id += '_lk' + ('1' if str(optimizer_kwargs["use_locking"]) == 'True' else '0')


        elif optimizer_name == 'ExtendedNesterovConst':
            _id += 'ENC'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
            _id += '_a' + str(optimizer_kwargs["alpha"])
            _id += '_b' + str(optimizer_kwargs["beta"])
            _id += '_lk' + ('1' if str(optimizer_kwargs["use_locking"]) == 'True' else '0')


        elif optimizer_name == 'AggregatedMomentum':
            _id += 'AgM'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
            _id += '_a' + str(optimizer_kwargs["a"])
            _id += '_K' + str(optimizer_kwargs["K"])
            _id += '_lk' + ('1' if str(optimizer_kwargs["use_locking"]) == 'True' else '0')


        elif optimizer_name == 'SSA1Const':
            _id += 'SSA1'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
            _id += '_b' + str(optimizer_kwargs["beta"])
            _id += '_k' + str(optimizer_kwargs["k"])
            _id += '_lk' + ('1' if str(optimizer_kwargs["use_locking"]) == 'True' else '0')


        elif optimizer_name == 'SSA2Const':
            _id += 'SSA2'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
            _id += '_b' + str(optimizer_kwargs["beta"])
            _id += '_k' + str(optimizer_kwargs["k"])
            _id += '_lk' + ('1' if str(optimizer_kwargs["use_locking"]) == 'True' else '0')


        elif optimizer_name == 'SSA2Nonconst':
            _id += 'SSA2NC'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
            _id += '_k' + str(optimizer_kwargs["k"])
            _id += '_a' + str(optimizer_kwargs["alpha"])
            _id += '_lk' + ('1' if str(optimizer_kwargs["use_locking"]) == 'True' else '0')


        elif optimizer_name == 'SSA1Nonconst':
            _id += 'SSA1NC'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
            _id += '_k' + str(optimizer_kwargs["k"])
            _id += '_a' + str(optimizer_kwargs["alpha"])
            _id += '_lk' + ('1' if str(optimizer_kwargs["use_locking"]) == 'True' else '0')


        elif optimizer_name == 'ExtendedNesterovNonconst':
            _id += 'ENNC'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
            _id += '_a' + str(optimizer_kwargs["alpha"])
            _id += '_b' + str(optimizer_kwargs["beta"])
            _id += '_g' + str(optimizer_kwargs["gamma"])
            _id += '_lk' + ('1' if str(optimizer_kwargs["use_locking"]) == 'True' else '0')


        # see kfac/kfac/python/ops/optimizer.py
        # seems to be buggy, we should write to the authors..
        # however PeriodicInvCovUpdateKfacOpt with invert_every=1 and cov_update_every=1
        # could be an alternative
        elif optimizer_name == 'KfacOptimizer':
            _id += 'KFAC'
            _id += TFOptimizers._get_kfac_parameters(optimizer_kwargs)

        # see kfac/kfac/python/ops/kfac_utils/periodic_inv_cov_update_kfac_opt.py
        elif optimizer_name == 'PeriodicInvCovUpdateKfacOpt':
            _id += 'PKFAC'
            _id += TFOptimizers._get_kfac_parameters(optimizer_kwargs)
            _id += '_ie' + str(optimizer_kwargs["invert_every"])
            _id += '_cue' + str(optimizer_kwargs["cov_update_every"])

        # see kfac/kfac/python/ops/kfac_utils/async_inv_cov_update_kfac_opt.py
        # (not of particular interests for the moment)
        elif optimizer_name == 'AsyncInvCovUpdateKfacOpt':
            _id += 'AKFAC'
            _id += TFOptimizers._get_kfac_parameters(optimizer_kwargs)

        elif optimizer_name == 'NaturalBackPropagationOptimizer':
            _id += 'NBP'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
            _id += '_d' + get_lr_id(optimizer_kwargs["damping"])
            _id += '_me' + str(optimizer_kwargs["memory_efficient"])

        elif optimizer_name == 'NaturalGradientOptimizer':
            _id += 'NG'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
            _id += '_d' + get_lr_id(optimizer_kwargs["damping"])

        elif optimizer_name == 'DropOneLogitGradientDescentOptimizer':
            _id += 'OLGD'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])

        elif optimizer_name == 'WakeSleepOptimizer':
            _id += 'WSO'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
            _id += ('_po' + optimizer_kwargs["post_optimizer"] if "post_optimizer" in optimizer_kwargs and optimizer_kwargs[
                    "post_optimizer"] is not None else '_poGD')

        elif optimizer_name == 'WakeSleepGradientDescentOptimizer':
            _id += 'WSGDO'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
            _id += ('_po' + optimizer_kwargs["post_optimizer"] if "post_optimizer" in optimizer_kwargs and optimizer_kwargs[
                    "post_optimizer"] is not None else '_poGD')

        elif optimizer_name == 'ReweightedWakeSleepOptimizer':
            _id += 'RWSO'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
            _id += ('_po' + optimizer_kwargs["post_optimizer"] if "post_optimizer" in optimizer_kwargs and optimizer_kwargs[
                    "post_optimizer"] is not None else '_poGD')
            _id += '_qb' + str(optimizer_kwargs["q_baseline"])

        elif optimizer_name == 'BidirectionalReweightedWakeSleepOptimizer':
            _id += 'BiO'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
            _id += (
                '_po' + optimizer_kwargs["post_optimizer"] if "post_optimizer" in optimizer_kwargs and optimizer_kwargs[
                    "post_optimizer"] is not None else '_poGD')

        elif optimizer_name == 'NaturalWakeSleepOptimizer':
            _id += 'NWSO'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
            _id += '_dp' + get_lr_id(optimizer_kwargs["diagonal_pad"])
            _id += ('_po' + optimizer_kwargs["post_optimizer"] if "post_optimizer" in optimizer_kwargs and optimizer_kwargs[
                    "post_optimizer"] is not None else '_poGD')

        elif optimizer_name == 'NaturalWakeSleepOptimizerAlternate':
            _id += 'NWSOAL'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
            _id += '_dp' + get_lr_id(optimizer_kwargs["diagonal_pad"])
            _id += ('_po' + optimizer_kwargs["post_optimizer"] if "post_optimizer" in optimizer_kwargs and optimizer_kwargs[
                    "post_optimizer"] is not None else '_poGD')
            _id += '_ks' + str(optimizer_kwargs["k_step_update"])

        elif optimizer_name == 'NaturalReweightedWakeSleepOptimizerAlternate':
            _id += 'NRWSOAL'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
            _id += '_dp' + get_lr_id(optimizer_kwargs["diagonal_pad"])
            _id += ('_po' + optimizer_kwargs["post_optimizer"] if "post_optimizer" in optimizer_kwargs and optimizer_kwargs[
                    "post_optimizer"] is not None else '_poGD')
            _id += '_ks' + str(optimizer_kwargs["k_step_update"])
            _id += '_qb' + str(optimizer_kwargs["q_baseline"])

        elif optimizer_name == 'NaturalBidirectionalOptimizer':
            _id += 'NBiDO'
            _id += '_lr' + get_lr_id(optimizer_kwargs["learning_rate"])
            _id += '_dp' + get_lr_id(optimizer_kwargs["diagonal_pad"])
            _id += ('_po' + optimizer_kwargs["post_optimizer"] if "post_optimizer" in optimizer_kwargs and optimizer_kwargs[
                    "post_optimizer"] is not None else '_poGD')
            _id += '_ks' + str(optimizer_kwargs["k_step_update"])
            _id += '_qb' + str(optimizer_kwargs["q_baseline"])
        else:
            raise Exception("training algorithm not recognized: %s ", optimizer_name)

        return _id
