import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from datasets.Dataset import TRAIN_LOOP, TRAIN, VALIDATION, TEST, \
                    TRAIN_SHUFFLED

import tensorflow as tf
import tensorflow_probability as tfp
from argo.core.TFDeepLearningModel import load_model_without_session
from argo.core.optimizers.LearningRates import process_learning_rate
from argo.core.utils.argo_utils import eval_method_from_tuple
from argo.core.network.KerasNetwork import KerasNetwork
from argo.core.hooks.LoggerHelperMultiDS import LoggerHelperMultiDS
from argo.core.flows.build_flow import build_flow
from argo.core.optimizers.TFOptimizers import TFOptimizers
from pprint import pprint

from tqdm import tqdm
import importlib


class Calibrator:
    def __init__(self, ffconffile, global_step, method, covcal, flow_params, optimizer_tuple, ci_name, alpha=0,
                 n_samples=1, sess=None, gpu=0, seed=100, base_dir='/data1/calibration'):

        self.sess = sess

        if sess is None:
            self.create_session(gpu=gpu, seed=seed)

        self._log_dir = self.create_log_dir_name(base_dir, ffconffile, method, covcal, alpha, flow_params, optimizer_tuple)

        self._optimizer_tuple = optimizer_tuple
        self._optimizer_name = optimizer_tuple[0]
        self._optimizer_kwargs = optimizer_tuple[1]
        self._n_samples = n_samples

        self._alpha_parameter = alpha
        self._use_alpha = (alpha!=0)

        self.method = method
        self.covcal = covcal

        self._flow_params = flow_params
        self._ffconffile = ffconffile
        self._global_step = global_step
        self._gpu = gpu
        self._seed = seed
        self._base_dir = base_dir

        # CONFIDENCE INTERVAL
        self._ci_class = ci_name


    def create_log_dir_name(self, base_dir, ffconffile, method, covcal, alpha_parameter, flow_params, optimizer_tuple):
        dsname = ffconffile.split('/')[-3]
        ffname = ffconffile.split('/')[-2]

        log_dir = os.path.join(base_dir, dsname, ffname, method, covcal)

        if flow_params is not None:
            log_dir += "-f" + flow_params['name']
            log_dir += "_n{:d}".format(flow_params['num_bijectors'])
            log_dir += "_hc{:d}".format(flow_params['hidden_channels'])

        log_dir += '_alpha_{}'.format(alpha_parameter)

        log_dir += '-tr' + TFOptimizers.create_id(optimizer_tuple)

        return log_dir

    def create_session(self, gpu=0, seed=0):
        # CREATE SESSION
        tf.set_random_seed(seed)
        # tf.reset_default_graph()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = '{:d}'.format(gpu)
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

    def release(self):
        self.sess.close()
        tf.reset_default_graph()

    def prepare_for_calibration(self, dataset_eval_names=[TRAIN, VALIDATION, TEST], dataset_ci_names=[VALIDATION, TEST], ci_period=30, **ci_kwargs):

        # LOAD THE MODEL BUT NOT THE MONITORED SESSION SINCE WE WANT TO CONTROL THE SESSION
        model, dataset, checkpoint_name = load_model_without_session(self._ffconffile, global_step=self._global_step, model_class_base_path="core")
        # RESTORE THE WEIGHTS
        saver = tf.train.Saver(None, max_to_keep=None, save_relative_paths=True)
        saver.restore(self.sess, checkpoint_name)

        # INPUTS
        #before any form of preprocessing and augmentation (if present)
        self.raw_x = model.raw_x

        #when you will have noise, use model._x before tiling, instead of model.raw_x
        # self.x = tf.tile(self.raw_x, [self._n_samples, 1, 1, 1])
        self.x = model.x
        self.y = model.y

        self.argo_model = model
        self._n_samples_ph = model.n_samples_ph

        # NETWORK
        network = model._network

        self.n_batches_per_epoch = model.n_batches_per_epoch
        self.is_training = model.is_training

        #update batch renorm
        update_batch_renorm = False
        if self.method=='finetune':
            update_batch_renorm = True

        self.extra_feed_kwargs_calibration = {self.is_training: update_batch_renorm} # consider setting this to false or not set it for calibrations which are not 'finetuning'
        # self.extra_feed_kwargs_calibration = {}

        wb_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        if isinstance(network, KerasNetwork):
            # FOR KERAS
            reg_losses, kl_losses, update_ops = model._network.get_keras_losses(model.x)

            self.total_reg = tf.add_n(wb_regularizers+reg_losses, name="regularization")
            self.total_KL = tf.reduce_sum(kl_losses) / model.dataset.n_samples_train
            self.update_ops = update_ops + tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            net_model = model._network._net_model
            distr_model = model._network._distr_model
            model_had_flow = model._network._flow is not None

        else:
            # FOR SONNET
            distr_model = model._network.module._distr_model
            net_model = None # TODO implement for sonnet

            self.total_reg = tf.add_n(wb_regularizers, name="regularization")
            self.total_KL = 0.
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        self.distr_model = distr_model
        self.net_model = net_model
        self.model_had_flow = model_had_flow
        if self.model_had_flow and self.covcal!="fix":
            raise Exception("If model had flows already, only fix covcal method is supported")

        self.calibration_tril = distr_model.calibration_tril
        self.calibration_tril_params = distr_model._calibration_tril_params


        self.prediction_distr = model.prediction_distr

        self.distr_for_calibration = None
        self.variables_for_calibration = None

        # FLOW
        flow_size = self.y.shape[-1]
        self._flow = None
        if self._flow_params is not None:
            self._flow = build_flow(self._flow_params, flow_size)

        # DATASET SECTION
        self.datasets_initializers = model.datasets_initializers
        self.ds_handle = model.ds_handle
        self.datasets_handles = self.sess.run(model.datasets_handles_nodes)
        self._parameters_list = model.dataset._parameters_list

        #here we will set the variables to optimize on and the distribution for which optimize the nll likelihood
        # we will set
        # self.distr_for_calibration
        # self.variables_for_calibration
        # self.logger with the tensors to log
        # tensor_nodes to log
        method = self.method
        covcal = self.covcal

        dtype = self.prediction_distr.dtype
        # bs is actually bs * ns, if a differentiation is needed the bs must be taken by raw_x before tiling
        bs = self.prediction_distr.batch_shape_tensor()[0]
        ps = self.prediction_distr.event_shape[0]

        calibration_variables_opt = []
        calibration_variables_init = []

        distr_for_calibration = None
        sc_cal = None
        distr_model = self.distr_model

        # huge if, it determines distr and vars for optimization (could be made more elegant with some classes...)
        if covcal=='scalar':
            calibration_scalar = tf.get_variable("calibration_scalar",
                                                 shape=(),
                                                 dtype=dtype,
                                                 initializer=tf.initializers.constant(1.))

            calibration_variables_init.append(calibration_scalar)
            calibration_variables_opt.append(calibration_scalar)

            if method=='mcsampling':
                mu_bar, sigma_bar = create_mu_bar_sigma_bar(self.x, self.argo_model, bs, ps, nsamples=10)

                distr_for_calibration = tfp.distributions.MultivariateNormalFullCovariance(loc=mu_bar,
                                                                       covariance_matrix=calibration_scalar*sigma_bar)

            else:

                model_loc = self.prediction_distr.parameters["loc"]
                model_scale_tril = self.prediction_distr.parameters["scale_tril"]

                distr_for_calibration = tfp.distributions.MultivariateNormalTriL(loc=model_loc, scale_tril=calibration_scalar*model_scale_tril)

                if method=='lastlayermu':
                    last_layer_weights = distr_model.dense_loc.weights
                    calibration_variables_opt+=last_layer_weights

                elif method=='lastlayer':
                    last_layer_weights = distr_model.dense_loc.weights + \
                                         distr_model.dense_diag_params.weights + \
                                         distr_model.dense_out_of_diag_params.weights

                    calibration_variables_opt+=last_layer_weights

                elif method == 'finetune':
                    last_layer_weights = distr_model.dense_loc.weights + \
                                         distr_model.dense_diag_params.weights + \
                                         distr_model.dense_out_of_diag_params.weights

                    calibration_variables_opt += last_layer_weights
                    calibration_variables_opt += self.net_model.trainable_variables

            sc_cal = calibration_scalar


        elif covcal=='tril' or covcal=='fix':
            # we need to create a new tril variable only for mcsampling method
            if method=='mcsampling':

                calibration_tril, calibration_tril_params = create_tril_var(ps, dtype)
                calibration_variables_init.append(calibration_tril_params)
                if covcal == 'tril':
                    calibration_variables_opt.append(calibration_tril_params)

                mu_bar, sigma_bar = create_mu_bar_sigma_bar(self.x, self.argo_model, bs, ps, nsamples=10)

                cal_sigma_bar = tf.einsum('ih,bhk,jk->bij', calibration_tril, sigma_bar, calibration_tril)

                distr_for_calibration = tfp.distributions.MultivariateNormalFullCovariance(loc=mu_bar,
                                                                       covariance_matrix=cal_sigma_bar)

            else:

                distr_for_calibration = self.prediction_distr

                calibration_tril = distr_model.calibration_tril

                if covcal=='tril':
                    # if it is `tril` I want to optimize these, if it is `fix`, no.
                    calibration_tril_params = distr_model._calibration_tril_params
                    calibration_variables_opt.append(calibration_tril_params)
                    calibration_variables_init.append(calibration_tril_params)

                if method=='lastlayermu':
                    last_layer_weights = distr_model.dense_loc.weights
                    calibration_variables_opt+=last_layer_weights

                elif method=='lastlayer':
                    last_layer_weights = distr_model.dense_loc.weights + \
                                         distr_model.dense_diag_params.weights + \
                                         distr_model.dense_out_of_diag_params.weights

                    calibration_variables_opt+=last_layer_weights

                elif method == 'finetune':
                    last_layer_weights = distr_model.dense_loc.weights + \
                                         distr_model.dense_diag_params.weights + \
                                         distr_model.dense_out_of_diag_params.weights

                    calibration_variables_opt += last_layer_weights
                    calibration_variables_opt += self.net_model.trainable_variables

                if self.model_had_flow:
                    print("\n FOUND FLOW I will recalibrate flow \n")
                    flow_name = model._network._flow_params['name']
                    if flow_name=="NVP":
                        flow_scope = "real_nvp_default_template"
                    elif flow_name in ["IAF", "MAF"]:
                        flow_scope = "masked_autoregressive_default_template"
                    else:
                        raise Exception("Don't know this flow: `{:}`  not supported".format(flow_name))

                    _flow_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=flow_scope)
                    
                    if not _flow_vars:
                        raise Exception("Could not find variables for model flow")

                    print("found flow variables:")
                    pprint(_flow_vars)
                    print("\n")

                    # don't reinitialize
                    calibration_variables_opt += _flow_vars

            sc_cal = tf.linalg.trace(calibration_tril) / tf.cast(ps, tf.float32)
            #np.save(log_dir+'calibration_tril.npy',calibration_tril)


        # TODO TEST create two optimizers, one for the "calibration handle" (scalar or tril) and another one for "other calibration variables"

        self.calibration_variables_init = calibration_variables_init
        self.calibration_variables_opt =  calibration_variables_opt#[n for n in calibration_variables_opt if not('calibration' in n.name)] # calibration_variables_opt

        # apply calibration flow if needed
        if self._flow is not None:
            if self.model_had_flow:
                raise Exception("model had flow already, apply additional flow is not yet supported.")

            print("\n ADDING FLOW I will calibrate with `{:}` flow\n".format(self._flow_params["name"]))
            pprint(self._flow_params)
            print("")

            with tf.variable_scope('flow'):
                distr_for_calibration = tfp.distributions.TransformedDistribution(
                                                                distribution=distr_for_calibration,
                                                                bijector=self._flow)
                #it is needed to instantiate the variables, because I need to collect variables to initialize the afterwards
                distr_for_calibration.sample(1)

            _flow_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='flow')
            print("found flow variables:")
            pprint(_flow_vars)
            print("\n")

            self.calibration_variables_init += _flow_vars
            self.calibration_variables_opt += _flow_vars

        self.distr_for_calibration = distr_for_calibration
        self.distr_sample = tf.squeeze(self.distr_for_calibration.sample(1), axis=0)

        # no need of these, pass directly distr_for_calibration to ConfidenceIntervals Class
        # self.sample_dis=self.distr_for_calibration.sample()
        # self.mean_dis=self.distr_for_calibration.mean()
        # self.var_dis=self.distr_for_calibration.variance()
        # self.covar_dis=self.distr_for_calibration.covariance()


        # DEFINE LOSS and OPTIMIZER
        # self._global_step_restore_int = sess.run(model.global_step)
        self.cal_global_step = tf.get_variable("cal_global_step",
                                            shape = (),
                                            dtype = tf.int64,
                                            trainable=False,
                                            initializer=tf.initializers.constant(0))

        lr = process_learning_rate(self._optimizer_kwargs["learning_rate"], self.cal_global_step)
        self._optimizer_kwargs["learning_rate"] = lr

        self.cal_loss, self.nll , self.training_op, self.optimizer, self.optimizer_variables = self.prepare_training(self.distr_for_calibration,
                                                                                                                     self.y,
                                                                                                                     self._optimizer_tuple,
                                                                                                                     self.cal_global_step,
                                                                                                                     self.calibration_variables_opt)

        self.sc_cal = sc_cal
        self.lr = lr
        tensors_nodes = [self.cal_loss, self.nll, sc_cal, lr]
        tensors_names = ["cal_loss", "nll", "sc_cal", "lr"]

        self.logger = LoggerHelperMultiDS(self._log_dir, "calibration",
                                        tensors_names,
                                        tensors_nodes,
                                        self.ds_handle,
                                        self.datasets_initializers,
                                        self.datasets_handles,
                                        dataset_eval_names)

        self.dataset_ci_names = dataset_ci_names
        self.ci_period = ci_period
        # create the nodes for sampling etcetera
        CI_Class = load_ConfidenceIntervals_Class(self._ci_class)
        self.ci_obj = CI_Class(self._log_dir,
                          self.datasets_initializers,
                          self.datasets_handles,
                          self.ds_handle,
                          self._n_samples_ph,
                          self.distr_sample,
                          self.raw_x,
                          self.x,
                          self.y,
                          self._parameters_list,
                          **ci_kwargs)


    def prepare_training(self, distr, y, optimizer_tuple, global_step, var_list):

        shaper = tf.shape(y)
        y_tile = tf.tile(y, [self._n_samples_ph, 1])
        nll_batch = -distr.log_prob(y_tile)

        # if self._use_alpha:
        #     if self._alpha_parameter !=0:
        #         loss_per_minibatch = tf.scalar_mul(self._alpha_parameter,distr.log_prob(y_tile))
        #         #import pdb; pdb.set_trace()
        #         loss_per_minibatch_reshaped=tf.reshape(loss_per_minibatch, (self._n_samples_ph, shaper[0]))
        #         loss_per_minibatch_avg=tf.reduce_logsumexp(loss_per_minibatch_reshaped,axis=0)
        #         loss_per_sample=tf.scalar_mul(-1./self._alpha_parameter,loss_per_minibatch_avg)
        #     else:
        #         loss_per_minibatch = nll_batch
        #         loss_per_minibatch_reshaped = tf.reshape(loss_per_minibatch, (self._n_samples_ph, shaper[0]))
        #         loss_per_sample=tf.reduce_mean(loss_per_minibatch_reshaped,axis=0)
        #
        # else:
        #     loss_per_sample = nll_batch

        if self._use_alpha and self._alpha_parameter !=0:
            loss_per_minibatch = tf.scalar_mul(self._alpha_parameter,distr.log_prob(y_tile))
            loss_per_minibatch_reshaped=tf.reshape(loss_per_minibatch, (self._n_samples_ph, shaper[0]))
            loss_per_minibatch_avg=tf.reduce_logsumexp(loss_per_minibatch_reshaped,axis=0)
            loss_per_sample=tf.scalar_mul(-1./self._alpha_parameter,loss_per_minibatch_avg)
        else:
            loss_per_minibatch = nll_batch
            loss_per_minibatch_reshaped = tf.reshape(loss_per_minibatch, (self._n_samples_ph, shaper[0]))
            loss_per_sample=tf.reduce_mean(loss_per_minibatch_reshaped,axis=0)

        cal_loss = tf.reduce_mean(loss_per_sample, name="cal_loss") + self.total_KL + self.total_reg

        nll = tf.reduce_mean(nll_batch, name="nll")
        #cal_nll = tf.reduce_mean(-distr.log_prob(y), name="nll")
        optimizer = eval_method_from_tuple(tf.train, optimizer_tuple)
        training_op = create_training_op(optimizer, cal_loss, global_step, var_list=var_list, update_ops=self.update_ops)
        optimizer_variables = optimizer.variables()
        return cal_loss, nll ,training_op, optimizer, optimizer_variables


    # def calibrate(self, nepochs = 30, calibration_datasets=[TRAIN_SHUFFLED], sess=None):
    def calibrate(self, nepochs=30, sess=None):

        if sess is None:
            sess = self.sess

        sess.run(tf.variables_initializer(self.optimizer_variables))
        sess.run(tf.variables_initializer([self.cal_global_step]+self.calibration_variables_init))


        #TRAINING LOOP FOR CALIBRATION
        # sess_extra_args = self.logger.get_sess_run_args()
        self.logger.reset(sess)

        print("calibrating: {:}".format(self._log_dir))
        for i in tqdm(range(nepochs), "calibrating", dynamic_ncols=True):

            # self.make_one_epoch_loop(self.training_op, [], extra_feed_kwargs = self.extra_feed_kwargs_calibration)
            self.make_one_epoch(self.training_op, [], TRAIN_SHUFFLED, extra_feed_kwargs = self.extra_feed_kwargs_calibration)

            # for DATASET_STR in calibration_datasets:
            #     self.make_one_epoch(self.training_op, [], DATASET_STR, extra_feed_kwargs = self.extra_feed_kwargs_calibration)

            #every n steps do mcmc
            self.logger.log(sess, i+1)
            self.logger.reset(sess)
            self.logger.plot_groupby('dataset', suffix="nll", x='epoch', y='nll')
            self.logger.plot_groupby('dataset', suffix="cal_loss", x='epoch', y='cal_loss')
            self.logger.plot(suffix='sc_cal', x='epoch', y='sc_cal')
            self.logger.plot(suffix='lr', x='epoch', y='lr')
            if (i+1)%self.ci_period==0:
                self.ci_obj.do_when_triggered(sess, self.dataset_ci_names, i+1)


    def make_one_epoch(self, training_op, nodes, DATASET_STR, extra_feed_kwargs={}, sess=None):
        if sess is None:
            sess=self.sess

        sess.run(self.datasets_initializers[DATASET_STR])
        # pbar = tqdm(desc='make one epoch')

        while True:
            try:
                _, nodes_np = sess.run([training_op,
                                            nodes],
                                            feed_dict={
                                                self.ds_handle : self.datasets_handles[DATASET_STR],
                                                self._n_samples_ph : self._n_samples,
                                                **extra_feed_kwargs
                                            }
                                           )
                # pbar.update(1)

            except tf.errors.OutOfRangeError:
                break

    def make_one_epoch_loop(self, training_op, nodes, extra_feed_kwargs={}, sess=None):
        if sess is None:
            sess=self.sess

        # pbar = tqdm(desc='make one epoch')
        for step in range(self.n_batches_per_epoch):
            _, nodes_np = sess.run([training_op,
                                        nodes],
                                        feed_dict={
                                            self.ds_handle : self.datasets_handles[TRAIN_LOOP],
                                            self._n_samples_ph : self._n_samples,
                                            **extra_feed_kwargs
                                        }
                                       )


def create_tril_var(output_size, dtype):
    output_size=output_size.value
    n_out_of_diag_elems = int(output_size * (output_size - 1) / 2)

    n_tril = int(n_out_of_diag_elems + output_size)
    calibration_tril_params = tf.get_variable("calibration_tril_params",
                                              shape=(n_tril,),
                                              dtype=dtype,
                                              initializer=tf.initializers.constant(value=1.))

    calibration_tril = tf.contrib.distributions.fill_triangular(calibration_tril_params,
                                                                name="calibration_tril")
    return calibration_tril, calibration_tril_params

def create_mu_bar_sigma_bar(x, model, bs, ps, nsamples=3):
    x_repeat = tf.tile(x, [nsamples, 1, 1, 1])
    repeated_distr = model._network(x_repeat, is_training=model.is_training)
    repeated_sample = tf.reshape(repeated_distr.sample(), [nsamples, bs, ps])
    mu_bar = tf.reduce_mean(repeated_sample, axis=0)
    sigma_bar = tfp.stats.covariance(repeated_sample, sample_axis=0, event_axis=-1)

    sigma_bar = tf.ensure_shape(sigma_bar, [None,ps,ps])

    return mu_bar, sigma_bar

def create_training_op(optimizer, total_loss, global_step, var_list=None, update_ops=[], clip_value=100):
    # 1st part of minimize: compute_gradient
    #TODO check var_list is what we expect
    grads_and_vars = optimizer.compute_gradients(total_loss, var_list=var_list)

    # clip gradients
    grads_and_vars_not_none = [(g, v) for (g, v) in grads_and_vars if g is not None]
    grads = [g for (g, v) in grads_and_vars_not_none]
    variables = [v for (g, v) in grads_and_vars_not_none]
    clipped_grads, global_norm = tf.clip_by_global_norm(grads, clip_value)
    clipped_grads_and_vars = [(clipped_grads[i], variables[i]) for i in range(len(grads))]

    # 2nd part of minimize: apply_gradient
    optimizer_step = optimizer.apply_gradients(clipped_grads_and_vars, global_step=global_step)

    update_ops = tf.group(*update_ops)
    training_op = tf.group(update_ops, optimizer_step)
    return training_op

def load_ConfidenceIntervals_Class(class_name):
    module = importlib.import_module(".hooks."+class_name, '.'.join(__name__.split('.')[:-1]))
    return getattr(module, class_name)