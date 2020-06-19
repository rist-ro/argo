import tensorflow as tf

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from datasets.Dataset import TRAIN_LOOP, VALIDATION, short_name_dataset
#from datasets.Dataset import  linestyle_dataset, check_dataset_keys_not_loop

from ..utils.argo_utils import create_reset_metric, compose_name

from ..argoLogging import get_logger

tf_logging = get_logger()

from .LoggingMeanTensorsHook import LoggingMeanTensorsHook, evaluate_over_dataset

import pdb

# the FisherMatrixHook expects the following from the NaturaGradientOptimizer
#self._invFisher_D
#self._invFisher_S
#self._invFisher_V
#self._n_weights_layers
        
class FisherMatrixHook(LoggingMeanTensorsHook):
    """
    Needed to Average certain variables every N steps
    """

    def __init__(self,
                 model,
                 #fileName,
                 dirName,
                 #tensors_to_average,
                 #tensors_to_average_names,
                 #tensors_to_average_plots,
                 period,
                 #tensorboard_dir,
                 #trigger_summaries,
                 #print_to_screen=False,
                 plot_offset=0,
                 train_loop_key=TRAIN_LOOP,
                 # datasets_keys=[VALIDATION],
                 time_reference="epochs"
                 ):


        fileName = "fisher_matrix"
        dirName = dirName + '/fisher'
        average_steps = model._get_steps(period, time_reference)
        self._model = model
        
        tensors_to_average = [
            [[model._optimizer._invFisher_D],
             [model._optimizer._invFisher_S],
             [model._optimizer._invFisher_V]
            ],
        ]

        tensors_to_average_names = [
            [["invFisher_D"],
             ["invFisher_S"],
             ["invFisher_V"]
             ],
        ]

        tensors_to_average_plots = [
            [{"fileName": "invFisher_D"},
             {"fileName": "invFisher_S"},
             {"fileName": "invFisher_V"}
             ]
        ]

        
        super(FisherMatrixHook, self).__init__(model,
                                               fileName,
                                               dirName,
                                               tensors_to_average,
                                               tensors_to_average_names,
                                               tensors_to_average_plots,
                                               average_steps = average_steps,
                                               tensorboard_dir = None,
                                               trigger_summaries = False,
                                               # trigger_plot = True,
                                               print_to_screen=False,
                                               plot_offset=0,
                                               train_loop_key=train_loop_key,
                                               datasets_keys=[], #dataset_keys,
                                               time_reference=time_reference
                 )

        # reset the default metric
        self._default_metric = tf.metrics.mean_tensor
        
        print("FisherMatrixHook has been enabled")

    def after_run(self, run_context, run_values):
        if self._trigged_for_step:
            tf_logging.info("trigger for FisherMatrixHook")
            
        super().after_run(run_context, run_values)
        
    def plot(self):
        for i, (tensors_vertical_panel, files_panel) in enumerate(zip(self._tensors_names,
                                                                      self._tensors_plots)):
            
            if len(tensors_vertical_panel) > 0:

                # here it start the vertical panel
                for j, (tensors_names_panel, file_save) in enumerate(zip(tensors_vertical_panel, files_panel)):
                    for dataset_str in self._datasets_keys:

                        filePath = self._dirName + '/' + file_save["fileName"] + '_' + self._time_reference_str[0] + str(self._time_ref).zfill(4) 
                        #d = self._tensors_values[dataset_str][i][j][0]
                        m = np.load(filePath + '.npy')
                        
                        plt.figure()
                        mean = np.mean(m, axis=1)
                        plt.hist(mean, bins = 100)
                        plt.savefig(filePath + '.png')
                        

                # invFisher_D
                filePath = self._dirName + '/invFisher_D_' + self._time_reference_str[0] + str(self._time_ref).zfill(4) 
                D = np.load(filePath + '.npy')

                # invFisher_V
                filePath = self._dirName + '/invFisher_V_' + self._time_reference_str[0] + str(self._time_ref).zfill(4)  
                V = np.load(filePath + '.npy')

                # invFisher_S
                filePath = self._dirName + '/invFisher_S_' + self._time_reference_str[0] + str(self._time_ref).zfill(4)  
                S = np.load(filePath + '.npy')

                
                # computing aggregated statistics
                ws = self._model._optimizer._n_weights_layers
                l = len(ws)
                VS = np.dot(V,S)

                #'''
                # full Fisher
                C = np.zeros((np.sum(ws), np.sum(ws)))

                sum_i = 0
                for ind_i, i in enumerate(ws):
                    sum_j = 0
                    for ind_j, j in enumerate(ws):
                        if ind_j>=ind_i:
                            #print(i,j)
                            M = np.dot(VS[sum_i:sum_i+i,:],V[sum_j:sum_j+j,:].T)
                            
                            if ind_i==ind_j:
                                # add to the diagonal without creating the diagonal matrix and summing
                                for k in range(i):
                                    # this is needed so that the diagonal does not count in the mean above
                                    M[k,k] = D[sum_i+k]
                                    
                            filePath = self._dirName + '/invFisher_' + str(ind_i) + '_' +  str(ind_j) + '_' + self._time_reference_str[0] + str(self._time_ref).zfill(4) 
                            M = -np.absolute(M)
                            #pdb.set_trace()
                            plt.imsave(filePath, M, cmap="gray")
                            
                            #print(M)
                            #M[0,:] = 0
                            #M[-1,:] = 0
                            #M[:,0] = 0
                            #M[:,-1] = 0
                        
                            #C[sum_i:sum_i+i,sum_j:sum_j+j] = M
                            #C[sum_j:sum_j+j,sum_i:sum_i+i] = M.T
                            #pdb.set_trace()
                        sum_j += j
                    sum_i += i

                '''
                plt.figure(figsize=(20, 20))
                plt.matshow(C, fignum=1, cmap=plt.cm.gray)
                #plt.xticks(range(l), df.columns, fontsize=14, rotation=45)
                #plt.yticks(range(l), df.columns, fontsize=14)
                #cb = plt.colorbar()
                #cb.ax.tick_params(labelsize=14)
                #plt.title('correlation matrix') #, fontsize=16);
                '''

                '''
                pdb.set_trace() 
                fig, axes = plt.subplots(1,1)
                fig.set_size_inches(20, 20)
                img = axes.imshow(C[-1000:,-1000:], cmap=plt.cm.gray)
                #img = axes.imshow(M, cmap=plt.cm.gray)
                print(C.shape)
                plt.colorbar(img)
                plt.tight_layout()
                
                filePath = self._dirName + '/invFisher_' + self._time_reference_str[0] + str(self._time_ref) 
                plt.savefig(filePath + '.png')
                '''


                ''' aggregated

                Cmean = np.zeros((l,l))
                Cstd = np.zeros((l,l))

                sum_i = 0
                for ind_i, i in enumerate(ws):
                    sum_j = 0
                    for ind_j, j in enumerate(ws):
                        if ind_j>=ind_i:
                            #print(i,j)
                            M = np.dot(VS[sum_i:sum_i+i,:],V[sum_j:sum_j+j,:].T)
                            if ind_i==ind_j:
                                # add to the diagonal without creating the diagonal matrix and summing
                                for k in range(i):
                                    # this is needed so that the diagonal does not count in the mean above
                                    M[k,k] = 0 #D[sum_i+k]
                            mean = np.mean(M)
                            #std = np.sqrt(np.mean(np.power(M,2) - mean**2))
                            # free memory
                            M = M[::2,::2]
                            std = np.std(M)
                            Cmean[ind_i,ind_j] = mean
                            Cmean[ind_j,ind_i] = mean
                            Cstd[ind_i,ind_j] = std
                            Cstd[ind_j,ind_i] = std
                            #pdb.set_trace()
                        sum_j += j
                    sum_i += i
                
                plt.figure() #figsize=(19, 15))
                plt.matshow(Cmean) # fignum=f.number)
                #plt.xticks(range(l), df.columns, fontsize=14, rotation=45)
                #plt.yticks(range(l), df.columns, fontsize=14)
                cb = plt.colorbar()
                #cb.ax.tick_params(labelsize=14)
                #plt.title('correlation matrix') #, fontsize=16);

                filePath = self._dirName + '/invFisher_layer_mean_' + self._time_reference_str[0] + str(self._time_ref) 
                plt.savefig(filePath + '.png')


                plt.figure() #figsize=(19, 15))
                plt.matshow(Cstd) # fignum=f.number)
                #plt.xticks(range(l), df.columns, fontsize=14, rotation=45)
                #plt.yticks(range(l), df.columns, fontsize=14)
                cb = plt.colorbar()
                #cb.ax.tick_params(labelsize=14)
                #plt.title('correlation matrix') #, fontsize=16);

                filePath = self._dirName + '/invFisher_layer_std_' + self._time_reference_str[0] + str(self._time_ref) 
                plt.savefig(filePath + '.png')
                '''
                
    def log_to_file_and_screen(self, log_to_screen=False):

        #firstLog = True

        for i, (tensors_vertical_panel, files_panel) in enumerate(zip(self._tensors_names,
                                                                      self._tensors_plots)):
            
            if len(tensors_vertical_panel) > 0:

                '''
                if firstLog:
                    time_ref_shortstr = self._time_reference_str[0]
                    logstring = "[" + time_ref_shortstr + " " + str(self._time_ref) + "]"
                else:
                    logstring = ""
                '''
                # here it start the vertical panel
                for j, (tensors_names_panel, file_save) in enumerate(zip(tensors_vertical_panel, files_panel)):
                    # log to file
                    #line = str(self._time_ref)

                    '''
                    for dataset_str in self._datasets_keys:
                        logstring += " ".join(
                            [" " + compose_name(name, short_name_dataset[dataset_str]) + " " + "%.4g" 
                             #for (name, mean) in zip(tensors_names_panel, self._tensors_values[dataset_str][i][j][0])])
                             for name in tensors_names_panel])
                    '''
                    for dataset_str in self._datasets_keys:
                        ####################
                        # notice the mean[0]
                        #line += "\t" + "\t".join(["%.5g" % str(mean[0]) for mean in self._tensors_values[dataset_str][i][oj]])

                        filePath = self._dirName + '/' + file_save["fileName"] + '_' + self._time_reference_str[0] + str(self._time_ref).zfill(4) 
                        d = self._tensors_values[dataset_str][i][j][0]
                        np.save(filePath, d)

                        filePath = self._dirName + '/' + file_save["fileName"] + '_' + 'step' + '_' + self._time_reference_str[0] + str(self._time_ref).zfill(4) 
                        d = self._tensors_values[dataset_str][i][j][0]
                        np.save(filePath, d)
                        
                        #for mean in self._tensors_values[dataset_str][i][j]:
                        #   line += "\t" 

                    #line += "\t" + "%.2f" % self._elapsed_secs

                    #line += "\n"
                    #self._log_to_file(line, file_plot)

                #logstring += "  (%.2fs)" % self._elapsed_secs

                # log to screen
                #if firstLog and log_to_screen:
                #    tf_logging.info(logstring)
                #    firstLog = False


    def _after_run(self, run_context, run_values):

        #pdb.set_trace()

        '''
        self._tensors_values[self.model._optimizer._invFisher_D] = run_context.session.run(
            self._mean_values[self._train_loop_key])

        # mean over the other dataset_keys
        for dataset_str in self._no_train_loop_datasets_keys:
            self._tensors_values[dataset_str] = evaluate_over_dataset(run_context.session,
                                                                      self._ds_handle,
                                                                      self._ds_initializers[dataset_str],
                                                                      self._ds_handles[dataset_str],
                                                                      self._mean_values[dataset_str])

        '''              
    def _create_or_open_files(self):
        pass

    def _reset_file(self, session):
        pass

    def end(self, session):
        pass
