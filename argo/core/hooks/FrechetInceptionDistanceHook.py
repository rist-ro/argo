
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

from .EveryNEpochsTFModelHook import EveryNEpochsTFModelHook

from ..utils.argo_utils import create_list_colors, create_reset_metric
from .LoggingMeanTensorsHook import evaluate_means_over_dataset #,evaluate_means_over_dataset_sample

import tensorflow as tf

from ..argoLogging import get_logger
tf_logging = get_logger()

from datasets.Dataset import TRAIN_SHUFFLED, VALIDATION_SHUFFLED

import numpy as np
import pdb
from scipy import linalg

NETWORK_NAME = "inception_v4"
INCEPTION_V4_GRAPH_DEF = "core/argo/networks/inception_v4.pb"
INPUT_TENSOR = "input:0" # ["?, 299, 299, 3]
OUTPUT_TENSOR = "InceptionV4/Logits/AvgPool_1a/AvgPool:0" # [?, 1, 1, 1536]
IMAGE_WIDTH = 299
IMAGE_HEIGHT = 299
# images should be in [-1, 1]

class FrechetInceptionDistanceHook(EveryNEpochsTFModelHook):
    """
    Hook computing the Frechet Inception Distance
    """
    
    def __init__(self,
                 model,
                 period,
                 time_reference,
                 n_batches,
                 dirName,
                 id = None,
                 pb = None,
                 input_tensor = None,
                 output_tensor = None,
                 datasets_keys = [TRAIN_SHUFFLED, VALIDATION_SHUFFLED],
                 plot_offset = 0
                 ):
        
        if pb or id or input_tensor or output_tensor:
            # make sure that if the user specifies a pb, then all field necessary are present
            assert(pb and id and input_tensor and output_tensor)
            self.inception_network = False
            self.input_tensor = input_tensor
            self.output_tensor = output_tensor
            self.pb = pb
            self.network_name = id
        else:
            # use the default inception network
            self.inception_network = True
            self.network_name = NETWORK_NAME

        self._batch_size_eval = model.batch_size["eval"]
        self._n_images = n_batches * self._batch_size_eval
        self._n_batches = n_batches 
        
        # fileName should be set before calling super()
        #self._fileHeader = '# frechet_inception_distance, network = ' + self.network_name + ', num images = ' + str(self._n_images) + '\n'
        #self._fileHeader += '# period \t fid_train \t fid_validation \t time_sec'

        log_str = "Create FrechetInceptionDistanceHook for " + str(self._n_images) + " samples, network " + self.network_name

        self._hook_name = "scores"
        dirName = dirName + '/' + self._hook_name
        
        super().__init__(model, period, time_reference, datasets_keys, dirName=dirName, log_str=log_str, plot_offset=plot_offset)

        # nodes to be computed and saved
        self._tensors_names = [[['FID']]]
        fileName = "frechet_inception_distance-" + self.network_name + "-num_images-" + str(self._n_images)
        self._tensors_plots = [[{'fileName': fileName,
                                 'logscale-y': 0}]]
        self._tensors_values = {} #[[self._fid]]
        self._fileName = "frechet_inception_distance-" + self.network_name
        
    def _compute_mean_cov_real_images_samples(self, features, dataset_str):

        #features_shape_besides_first = features.shape.as_list()[1:]
        #features = tf.reshape(features, [self._batch_size_eval] + features_shape_besides_first)
        batch_size = tf.cast(tf.shape(features)[0], dtype=tf.float32)
        reduced_features = tf.reduce_mean(features, axis=0)
        dim_features = features.shape.as_list()[1]

        # check I have enough images to then invert the covariance matrix
        assert(self._n_images>dim_features, "The number of images used in the computation of the FrechetInceptionDistance (n_batches * self._batch_size_eval) must be great than the number of features of the classification network")
        
        mean_values, mean_update_ops, mean_reset_ops = create_reset_metric(tf.metrics.mean_tensor,
                                                                           scope=dataset_str + "_mean_reset_metric/" + features.name,
                                                                           values=reduced_features,
                                                                           weights=batch_size)
        
        
        cov_update_ops = []
        cov_reset_ops = []
        # self._cov_values=[]
        # init variables
        sum_w = tf.Variable(0.)
        x_n = tf.Variable(tf.zeros([dim_features], dtype=tf.float32))
        y_n = tf.Variable(tf.zeros([dim_features], dtype=tf.float32))
        C_n = tf.Variable(tf.zeros([dim_features, dim_features], dtype=tf.float32))
        # update ops
        add_w = tf.assign(sum_w, sum_w + batch_size)
        cov_update_ops.append(add_w)
        with tf.control_dependencies([add_w]):
            x_delta = tf.reduce_sum(features-x_n, axis=0) / add_w
            add_x_n = tf.assign(x_n, x_n + x_delta)
            cov_update_ops.append(add_x_n)

            with tf.control_dependencies([add_x_n]):
                C_delta = tf.matmul(tf.expand_dims(features - x_n, 0),
                                    tf.expand_dims(features - y_n, 0),
                                    transpose_a = True)
                add_C_n = tf.assign(C_n, C_n + tf.squeeze(C_delta,0))                        
                cov_update_ops.append(add_C_n)
                            
                with tf.control_dependencies([add_C_n]):
                    y_delta = tf.reduce_sum(features - y_n, axis=0) / add_w
                    add_y_n = tf.assign(y_n, y_n + y_delta)
                    cov_update_ops.append(add_y_n)
                            
        # reset ops
        cov_reset_ops.append(tf.assign(sum_w, 0))
        cov_reset_ops.append(tf.assign(x_n, tf.zeros([dim_features])))
        cov_reset_ops.append(tf.assign(y_n, tf.zeros([dim_features])))
        cov_reset_ops.append(tf.assign(C_n, tf.zeros([dim_features, dim_features])))
        
        cov_values = C_n / sum_w

        return mean_values, mean_update_ops, mean_reset_ops, cov_update_ops, cov_reset_ops, cov_values
    
    def _begin_once(self):
                
        x = self._model.x
        samples = self._model.samples
        #self.samples = tf.placeholder(x.dtype, shape=x.get_shape(), name='random_samples')

        self._mean_values_real = {}
        self._mean_update_ops_real = {}
        self._mean_reset_ops_real = {}

        self._cov_values_real = {}
        self._cov_update_ops_real = {}
        self._cov_reset_ops_real = {}

        self._mean_values_sample = {}
        self._mean_update_ops_sample = {}
        self._mean_reset_ops_sample = {}

        self._cov_values_sample = {}
        self._cov_update_ops_sample = {}
        self._cov_reset_ops_sample = {}

        if self.inception_network:
            
            graph_def = tf.contrib.gan.eval.get_graph_def_from_disk(INCEPTION_V4_GRAPH_DEF)
            network = lambda x: tf.contrib.gan.eval.run_image_classifier(x, graph_def, INPUT_TENSOR, OUTPUT_TENSOR)

            # check number of channales
            shape = tf.shape(x)
            channels = shape[-1]
            len_shape = len(x.shape.as_list())

            # make sure we have images with 3 channels
            x_3ch = tf.cond(tf.equal(channels,1), lambda: tf.tile(x, [1]*(len_shape-1) + [3]), lambda: x)
            samples_3ch = tf.cond(tf.equal(channels,1), lambda: tf.tile(samples, [1]*(len_shape-1) + [3]), lambda: samples)
        
            #samples_3channels = tf.expand_dims(samples_3channels[0],axis=0)
            #x_3channels = tf.expand_dims(x_3channels[0],axis=0)

            # resize images
            x_3ch = tf.image.resize_bilinear(x_3ch, [IMAGE_WIDTH, IMAGE_HEIGHT], align_corners=False)
            samples_3ch = tf.image.resize_bilinear(samples_3ch, [IMAGE_WIDTH, IMAGE_HEIGHT], align_corners=False)

            features_real = network(x_3ch)
            features_samples = network(samples_3ch)
            # more two extra dims
            features_real = tf.squeeze(features_real, [1,2])
            features_sample = tf.squeeze(features_samples, [1,2])

        else:
            
            graph_def = tf.contrib.gan.eval.get_graph_def_from_disk(self.pb)
            network = lambda x: tf.contrib.gan.eval.run_image_classifier(x,
                                                                         graph_def,
                                                                         self.input_tensor,
                                                                         self.output_tensor)

            features_real = network(x)
            features_sample = network(samples)
            
        for dataset_str in self._datasets_keys:
                
            self._mean_values_real[dataset_str], self._mean_update_ops_real[dataset_str], self._mean_reset_ops_real[dataset_str], self._cov_update_ops_real[dataset_str], self._cov_reset_ops_real[dataset_str], self._cov_values_real[dataset_str] = self._compute_mean_cov_real_images_samples(features_real, dataset_str)

            self._mean_values_sample[dataset_str], self._mean_update_ops_sample[dataset_str], self._mean_reset_ops_sample[dataset_str], self._cov_update_ops_sample[dataset_str], self._cov_reset_ops_sample[dataset_str], self._cov_values_sample[dataset_str] = self._compute_mean_cov_real_images_samples(features_sample, dataset_str) 
                               
        
    def do_when_triggered(self, run_context, run_values):
        tf_logging.info("trigger FrechetInceptionDistanceHook for " + str(self._n_images) + " samples, network " + self.network_name)
        #self.random_samples = self._model.generate(batch_size=self._n_batches)

        real_images_means = {}
        real_images_covs = {}
        sample_means = {}
        sample_covs = {}
        
        # mean over the other dataset_keys
        for dataset_str in self._datasets_keys:

            real_images_means[dataset_str], sample_means[dataset_str] = evaluate_means_over_dataset(run_context.session,
                                                                                                    self._ds_handle,
                                                                                                    self._ds_initializers[dataset_str],
                                                                                                    self._ds_handles[dataset_str],
                                                                                                    [self._mean_values_real[dataset_str], self._mean_values_sample[dataset_str]],
                                                                                                    [self._mean_update_ops_real[dataset_str], self._mean_update_ops_sample[dataset_str]],
                                                                                                    [self._mean_reset_ops_real[dataset_str], self._mean_reset_ops_sample[dataset_str]],
                                                                                                    max_iterations = self._n_batches)

        
            real_images_covs[dataset_str], sample_covs[dataset_str]= evaluate_means_over_dataset(run_context.session,
                                                                                                 self._ds_handle,
                                                                                                 self._ds_initializers[dataset_str],
                                                                                                 self._ds_handles[dataset_str],
                                                                                                 [self._cov_values_real[dataset_str], self._cov_values_sample[dataset_str]],
                                                                                                 [self._cov_update_ops_real[dataset_str], self._cov_update_ops_sample[dataset_str]],
                                                                                                 [self._cov_reset_ops_real[dataset_str], self._cov_reset_ops_sample[dataset_str]],
                                                                                                 max_iterations = self._n_batches)


            try:
                
                # compute fid
                diff = real_images_means[dataset_str] - sample_means[dataset_str]

                # product might be almost singular
                covmean, _ = linalg.sqrtm(real_images_covs[dataset_str].dot(sample_covs[dataset_str]), disp=False)
                if not np.isfinite(covmean).all():
                    msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
                    warnings.warn(msg)
                    offset = np.eye(real_images_covs[dataset_str].shape[0]) * eps
                    covmean = linalg.sqrtm((real_images_covs[dataset_str] + offset).dot(sample_covs[dataset_str] + offset))
                
                # numerical error might give slight imaginary component
                if np.iscomplexobj(covmean):
                    if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                        m = np.max(np.abs(covmean.imag))
                        raise ValueError("Imaginary component {}".format(m))
                    
                    covmean = covmean.real

                tr_covmean = np.trace(covmean)

                fid = diff.dot(diff) + np.trace(real_images_covs[dataset_str]) + np.trace(sample_covs[dataset_str]) - 2 * tr_covmean

            except ValueError:
                print("Error in computing the FrechetInceptionDistance, most likely nunrical issues in computing the sqrt of a convariance matrix. Try increasing the number of samples")
                fid = -1.

                
            self._tensors_values[dataset_str] = [[[fid]]]

        self.log_to_file_and_screen()

    '''
    def plot(self):
        
        # create figure
        fig = plt.figure(figsize=(20, 9))
        fig.suptitle(self._model.id, y=0.995, fontsize=10)

        filePath = self._dirName + '/' + self._fileName + '.txt'
                    
        # read data from file
        with open(filePath) as f:
            data = f.read()

        data = data.split('\n')

        first_line = data[0]
        while first_line[0]=="#":
            data = data[1:]
            first_line = data[0]
        
        offset = 0
        data = data[offset:-1]
        x = [int(row.split("\t")[0]) for row in data]

        ax = fig.add_subplot(1, 1, 1)

        # frechet_inception_distance_train
        y = [float(row.split("\t")[1]) for row in data]
        ax.plot(x, y, linestyle="-", c='b', label="train")

        # frechet_inception_distance_evaluation
        y = [float(row.split("\t")[2]) for row in data]
        ax.plot(x, y, linestyle="--", c='b', label="test")
            
        ax.set_xlabel(self._time_reference_str)
        ax.set_ylabel("frechet_inception_distance")
       
        ax.legend()

        #ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        #ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))

        #xt = ax.get_xticks()
        #xt = np.append(xt, offset)
        #xtl = xt.astype(int).tolist()
        #xtl[-1]= offset
        #ax.set_xticks(xt)
        #ax.set_xticklabels(xtl)

        ax.grid()

        plt.tight_layout()

        plt.savefig(self._dirName + "/" + self._fileName + ".png")
        plt.close()
   
   '''
