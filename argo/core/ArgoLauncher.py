from abc import ABC, abstractmethod
import copy
import time
from time import gmtime, strftime

import gc
import os

import multiprocessing
from itertools import product
import subprocess
import numpy as np

from datasets.Dataset import Dataset

from .utils.argo_utils import load_class, make_list
import traceback
import pdb
import pprint

def get_full_id(dataset, launchable):
    """ This is the full id of an experiment. As a convention, the dataset is specifying a folder and
    the launchable object is specifying the name of the model used for training.
    """
    return dataset.id + '/' + launchable.id

# this is the class of the consumer
class ProcessTask(multiprocessing.Process):

    def __init__(self, tasks_queue, gpu_allocation_queue, launcher): #, return_queue
        super(ProcessTask, self).__init__()

        print("Creating consumer " + str(os.getpid()))

        self.tasks_queue = tasks_queue
        self.gpu_allocation_queue = gpu_allocation_queue
        self.lock = None
        self.dependencies = None
        self.launcher = launcher

    def set_dependencies(self, dependencies, lock):
        self.dependencies = dependencies
        self.lock = lock

    def run(self):
        while True:
            # this accessed in a safe way by the ProcessTask
            task_and_config = self.tasks_queue.get()

            if task_and_config is None:
                # Poison pill means shutdown
                print("Exiting from consumer " + str(os.getpid()))
                self.tasks_queue.task_done()
                break

            (task, config) = task_and_config

            gpu =  self.gpu_allocation_queue.get()
            print("Running on GPU " + str(gpu))

            executed = False
            #import pdb;pdb.set_trace()
            try:
                executed = self.launcher.task_execute(task, config, gpu,
                                                      self.dependencies,
                                                      self.lock,
                                                      "Consumer " + str(os.getpid()))
            except Exception as exc:
                errfile = self.launcher._launchable.dirName + '/error.log'

                errstream = open(errfile, 'a')
                errtime = strftime("%Y-%m-%d %H:%M:%S\n", gmtime())
                errstream.write("\nError occurred at: " + errtime)
                errstream.write("Failed to execute job: \n"+ str(exc) + "\n")
                trace = traceback.format_exc()
                errstream.write(trace)
                errstream.close()

                print("Failed to execute job: \n"+ str(exc) + "\n")

            self.gpu_allocation_queue.put(gpu)
            self.tasks_queue.task_done()
            print("Consumer PID = " + str(os.getpid()) + " has processed the job")

            # #TODO why this? if you failed there was a reason.. no? Temporarily commented failure policy
            # if not executed:
            #     # after 10s, put back the task in the queue
            #     time.sleep(10)
            #     self.tasks_queue.put(task_and_config)
            #     print("Consumer PID = " + str(os.getpid()) + " has put back the job in the queue")
            # else:
            #     self.tasks_queue.task_done()
            #     print("Consumer PID = " + str(os.getpid()) + " has processed the job")

# this is the class of the consumer when we run on distributed envorinments, such as the UBB cluster
class ProcessDistributedTask(multiprocessing.Process):

    def __init__(self, tasks_queue, gpu_allocation_queue, launcher, node_number, ip): #, return_queue
        super(ProcessDistributedTask, self).__init__()

        print("Creating consumer from " + str(os.getpid()))

        self.tasks_queue = tasks_queue
        self.gpu_allocation_queue = gpu_allocation_queue
        #self.return_queue = return_queue

        #self.task_execute = task_execute
        self.launcher = launcher

        self.node_number = node_number
        self.ip = ip

    def run(self):
        while True:
            task_and_config = self.tasks_queue.get()

            if task_and_config is None:
                # Poison pill means shutdown
                print("Exiting from consumer " + str(os.getpid()))
                self.tasks_queue.task_done()
                break

            (task, config) = task_and_config

            gpu =  self.gpu_allocation_queue.get()
            print("Planned to run on GPU " + str(gpu) + " at " + self.ip)

            # replace in config specific inforation about the node/GPU
            config["nodes"] = [config["nodes"][self.node_number]]
            config["nodes"][0]["used_GPUs"] = {gpu}
            config["nodes"][0]["cores_per_GPU"] = 1

            #self.task_execute(task, config, gpu, "Consumer " + str(os.getpid()))
            self.launcher.task_execute_remote(task, config, gpu, self.ip, "Consumer " + str(os.getpid()))
            self.gpu_allocation_queue.put(gpu)

            self.tasks_queue.task_done()
            print("Consumer done PID = " + str(os.getpid()))

# the main launcher
class ArgoLauncher(ABC):


    @abstractmethod
    def execute(self, model, dataset, opts, config):
        raise Exception("Implement execute() in your launcher")

    def initialize(self, launchable, config):
        """For more complicated operations (of the Launcher) that require some preinitialization.
        """
        pass


    def _load_model_class(self, class_name):
        try:
            # try to get the module from core
            load_model_class = load_class(class_name, base_path="core")

        except Exception as e:
            raise Exception("problem loading model: %s, exception: %s" % (class_name, e)) from e

        return load_model_class

    def lock_resources(self, lock, dependencies, task_opts, task_config):
        # nothing to be done, this function should be overwritten by child Launchers
        #
        # start of safe zone
        #lock.acquire()
        # end of safe zone
        #lock.release()
        return True

    def unlock_resources(self, lock, dependencies, task_opts, task_config):
        # nothing to be done, this function should be overwritten by child Launchers
        #
        # start of safe zone
        #lock.acquire()
        # end of safe zone
        #lock.release()
        pass

    def task_execute(self, dm_params, config, gpu=-1, dependencies=None, lock=None, message_prefix=""):
        """
        this method takes care of executing a task passed as a parameter
        the task is defined in opts, while config contains other information necessary to
        run the task, including information about loggers

        Args:
            config:
            gpu:
            dependencies:
            lock:
            message_prefix:

        Returns:

        """

        # There is a reason to import tensorflow only here, but I don't remember it
        # since I am a responsible person, I provide you a link to satisfy your curiosity
        # see https://zhuanlan.zhihu.com/p/24311810
        # (oppps unfortunately it's in Chinese ;-( )
        # Since I am extremely nice (and feel guilty) I provide a translation
        #
        #It should be noted that some side effects occur when Cuda tools such as import theano
        #or import tensorflow are called, and the side effects are copied to the child processes
        #as they are and errors then occur, such as:
        #
        #     could not retrieve CUDA device count: CUDA_ERROR_NOT_INITIALIZED
        #
        # The solution is to ensure that the parent process does not introduce these tools,
        # but after the child process is created, let the child process each introduced.

        import tensorflow as tf
        # I have to reset the graph here, because there might be leftover from previous jobs
        # it happens if you use tf.variable_scope(. ,reuse=None) somewhere you see the error
        tf.reset_default_graph()

        dataset_params, model_params = dm_params

        #TODO this has to be fixed when the optimization module will be created
        dataset = Dataset.load_dataset(dataset_params)

        # I need to copy configs here in case some algorithms are adding stuffs to the dictionaries,
        # e.g. regularizers etcetera... Since python is passed by reference they get modified before writing
        # to file resulting in unreadable confs from experiment folders. Please leave it here (Riccardo)
        model_params_orig = copy.deepcopy(model_params)
        dataset_params_orig = copy.deepcopy(dataset_params)
        config_orig = copy.deepcopy(config)

        # this messages_prefix is used to debug purposes, in particular to distinguish the prints
        # coming from different consumers. It is a prefix added to each print
        if message_prefix != "":
            message_prefix += " "

        # setting the seed for numpy for the task, which is specified in opts["seed"], specified
        # set in the function create_opts_list where the Cartesian product is computed
        print(message_prefix + "setting seed=" + str(model_params["seed"]))
        np.random.seed(model_params["seed"])

        # create the full_id, which includes
        # ALGORITHM-NAME_DATASET-NAME-WITH-OPTIONS_ALGORITHM-OPTIONS
        # notice that this method may be overwritten by some Launchers, such as
        # TestAdvExamplesLauncher, which implements more sophisticated naming convetions for the
        # algorithm

        # get the class to load
        launchableClass = self._load_model_class(model_params["model"])

        # add information about the dataset for the launchable construction, needed in view of future keras compatibility
        # try catch to allow compatibility for datasets which do not have labels (see Dataset interface)
        try:
            output_shape = dataset.y_shape
        except ValueError:
            output_shape = None

        dataset_info = {"output_shape" : output_shape,
                        "input_shape" : dataset.x_shape_train}

        model_params.update(dataset_info)

        baseDir = config["dirName"]+"/"+dataset.id

        # TODO why check_ops is so high in the hierarchy?
        check_ops = getattr(config, "check_ops", False)

        self._launchable = launchableClass(model_params, baseDir, check_ops=check_ops, gpu=gpu, seed=model_params['seed'])


        dirName = self._launchable.dirName
        full_id = get_full_id(dataset, self._launchable)

        print(message_prefix + "got a new job, checking " + full_id)

        # check if the algorithm has been executed previously and if successfully completed
        # this is certified by the existence of a log file in dirName
        log_file = dirName + '/experiment.log'

        if not os.path.isfile(log_file):

            # if not, I need to prepare to execute the algorithm, by first creating the necessary
            # directories, such as those to save modes and log general purpose quantities, in case
            # this is specified in the config

            # if lock is None, there is no need to lock_resouces, since there is not parallelism
            if lock:
                print(message_prefix + "consumer " + str(os.getpid()) + " checking locks for " + full_id)
                lock_resources = self.lock_resources(lock, dependencies, model_params, config)
            else:
                lock_resources = True

            # in case lock_resoucers is false, then I cannot run the algorithm, thus I return false
            if not lock_resources:
                print(message_prefix + "consumer " + str(os.getpid()) + " lock not available " + full_id)
                return False
            else:
                print(message_prefix + "consumer " + str(os.getpid()) + " available or locked " + full_id)

            # create directories where the conf and txt files are saved, notice that in case of more sophisticated algorithms
            # the function can be overwritten by the child Launcher, as in TestAdvExamplesLauncher
            # dirName, launchable_id = self.create_path_directories(path)
            os.makedirs(dirName, exist_ok=True)

            # choose between running on GPU or CPU
            if gpu == -1:
                print(message_prefix + "running on cpu")
                device = '/cpu:0'
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
                print(message_prefix + "running on gpu = " + str(gpu))
                device = '/gpu:' + '0'

            # create a single conf file which allows to execute only the specific instance extracted
            # by the Cartesian product, independently from the contents of the original conf file
            ArgoLauncher.write_conf_file(dirName + '/experiment.conf',
                                         dataset_params_orig,
                                         model_params_orig,
                                         config_orig)

            # instantiate algorithm to be run

            self.initialize(self._launchable, config)

            # start timer
            start = strftime("%Y-%m-%d %H:%M:%S\n", gmtime())
            startTime = time.time()
            assert(self._launchable is not None), "the Launchable object has to be instantiated before executing"

            self.execute(self._launchable, dataset, model_params, config)

            # stop timer
            end = strftime("%Y-%m-%d %H:%M:%S\n", gmtime())
            endTime = time.time()

            self._launchable.release()
            del dataset
            gc.collect()
            print("Released")

            print(message_prefix + "consumer " + str(os.getpid()) + " unlocking resources for " + full_id)
            if lock:
                self.unlock_resources(lock, dependencies, model_params, config)


            f = open(log_file, 'w')
            f.write("started at " + start)
            f.write("done at " + end)
            f.write("duration " + str(endTime - startTime) + "\n")
            f.write("seed used is " + str(model_params["seed"]) + "\n")
            f.write("gpu " + str(gpu) + "\n")

            f.close()

            print(message_prefix + "completed job " + full_id)
        else:
            print(message_prefix + "found completed job " + full_id)


        return True


    @classmethod
    def process_args(cls, argv):
        usage_message = "Usage: python3 ArgoLauncher.py [paramFile.conf]" \
                        " [single|pool|distributed|stats]"
        if len(argv)<3:
            raise Exception(usage_message)

        file_name_path = argv[1]

        if argv[2]=="single":
            parallelism = 0
        elif argv[2]=="pool":
            parallelism = 1
        elif argv[2]=="distributed":
            parallelism = 2
        elif argv[2]=="stats":
            parallelism = -1
        else:
            raise Exception(usage_message)

        return cls.process_conf_file(file_name_path) + (parallelism, ) #network, ip)

    @staticmethod
    def process_conf_file(file_name_path):
        """Read from a config file

        Args:
            file_name_path (string): path to the config file.

        Returns:
            dict: parameters for the initialization of the Dataset class
            dict: parameters for the algorithm (in list form, not cartesian product yet)
            dict: parameters related to the execution of the processes (machine architecture, gpus, etc..)

        """
        if not os.path.isfile(file_name_path):
            raise Exception('File "' + file_name_path + "' does not exists\n")
        with open(file_name_path,'r') as fstream:
            # load the three dictionaries
            dataset_params, launchable_params, config = eval(fstream.read())


            np.random.seed(config["seed"])
            config["seeds"] = np.random.randint(1, 10000, config["runs"]).tolist()

        return dataset_params, launchable_params, config

    #TODO-ARGO2 WHAT ARE OPTS, let us find a better name. dataset_params, model_lists_of_params, config
    #TODO-ARGO2 model_lists_of_params will be unpacked withcartesian product in a list of model_params
    @staticmethod
    def create_opts_list(dataset_params, model_params, config):
        """
        Check if the key run exist in model_params.
        If not compute all possible combinations in the model_params (cartesian product of lists of parameters).

        Args:
            dataset_params (dict): parameters of the Dataset.
            model_params (dict): Parameters for the LaunchableClass.
            config (dict): Part of the config file related to the execution (on this one,
                            no cartesian product will be done).

        Returns:
            list of dicts : list of launchable parameters
            list of dicts : list of config for the main algorithm and all the sub-algorithms, if any.

        """

        model_params_list = [dict(zip(model_params.keys(), p)) for p in product(*map(make_list, model_params.values()))]
        dataset_params_list = [dict(zip(dataset_params.keys(), p)) for p in product(*map(make_list, dataset_params.values()))]

        dmopts = list(product(dataset_params_list, model_params_list))
        # In case "run" is among the keys, this is the case in which the *.conf constains a run entry,
        # which implies the conf used is a saved conf from argo, which refers to a specific
        # run of out the runs
        if "run" in model_params.keys():
            return dmopts, config

        multiple_dmopts_list = []
        for r in range(config["runs"]):
            multiple_dmopts_list = multiple_dmopts_list + [copy.deepcopy(l) for l in dmopts]

        len_list = len(dmopts)
        # NB: this is not the way you do things, since dictionarie share references
        # multiple_dmopts_list = model_params_list * config["runs"]


        for r in range(config["runs"]):
            for i in range(len_list):
                index = len_list*r + i
                # I add seed and run to the model_params, leaving dataset_params untouched
                multiple_dmopts_list[index][1]["seed"] = config["seeds"][r]
                multiple_dmopts_list[index][1]["run"] = r

        return multiple_dmopts_list, config

    @staticmethod
    def write_conf_file(outputname, dataset_params, model_params, config):
        f = open(outputname, 'w')
        f.write("[\n")
        f.write(pprint.pformat(dataset_params))
        f.write(",\n")
        f.write(pprint.pformat(model_params))
        f.write(",\n")
        f.write(pprint.pformat(config))
        f.write("\n]")
        f.close()

    # this is the main method called to start the execution
    # consider this the entry point to understand the code
    def run(self, dataset_params, model_params, config, parallelism):
        """
        this is the main method called to start the execution
        consider this the entry point to understand the code

        1) The launcher creates a queue of tasks which depends on the conf file
        2) The launcher creates a pool of consumers. A consumer is a Process
            whose role is to take a task from the list and execute it if
            necessary. Each consumer work on a GPU. Multiple consumers can work
            on the same GPU if cores_per_GPU > 1


        Args:
            dataset: the dataset used in the experiments, notice that is
                required only to run the experiment, but not to create the
                ArgoLauncher object, that's why it does not appear in the
                constructor of ArgoLauncher
            model_params: (dict)
            config: (dict)
            parallelism: (0,1,2)
                parallelism = 0, a single process is run on the first GPU from
                (single)         the list of used_GPUs in the conf file for
                                 nodes[0], the process is run on localhost
                                 (independently from nodes[0]["IP"])
                parallelism = 1, a pool of consumers (i.e., processes) is created, of size equal to
                (pool)           len(used_GPUs)*cores_per_GPU, all processes are run on localhost
                                 (independently from nodes[0]["ID"])
                parallelism = 2  a pool of processes is created on localhost, with consumers associated
                (distributed)    to each node in the list "nodes" from the conf file, the size of the pool
                                 equal the sum over the nodes i, of len(used_GPUs of node i)*cores_per_GPU
                                 of node i. All nodes are supposed to write in the same folder, which must
                                 be accessible from localhost. Refer to the code in ProcessDistributedTask,
                                 for details about how the communication between the node is implemented
                                 (hint: basically no communication is required, nodes are just slaves)


        """

        # default size of the pool of consumers
        poolSize = 1

        # if parallelism is 1, then I need to create a pool of processes to execute the tasks
        if parallelism==1:

            # I need to manage two queues, of for the tasks to be executed,
            # and one for the available GPUs, this is due to the fact that each consumer has to pick
            # up an available GPU where the task is run

            # TODO
            # I THINK THIS IS WRONG
            # In principle (most likely) the queue of GPUs is not required and we could initialize the
            # consumer with a redefined GPU

            # queue for the tasks, which is empty at the moment
            tasks_queue = multiprocessing.JoinableQueue()
            # queue for the GPU to be allocated
            gpu_allocation_queue = multiprocessing.JoinableQueue()

            # compute new pool size
            poolSize = len(config["nodes"][0]["used_GPUs"])*config["nodes"][0]["cores_per_GPU"]

            # add to the pool the available GPUs
            for i in range(config["nodes"][0]["cores_per_GPU"]):
                for j in config["nodes"][0]["used_GPUs"]:
                    gpu_allocation_queue.put(j)

            # here I create a lock used by the ProcessTask to access a dictionary of dependences of each algorithm,
            # which is used to avoid that multiple algorithm running in parallel could train the same dependence in case
            # it has not been trained before. This is the for instance the behavior of TestAdvExamplesLauncher.
            lock = multiprocessing.Lock()
            manager = multiprocessing.Manager()
            dependencies = manager.dict()

            # here I create the pool of executors, of size poolSize
            for k in range(poolSize):
                # it is ok to pass self to the processtask, since the message passing protocol use pickle (I guess) else use launcher=deepcopy(self)
                #TODO-ARGO2: why the launcher itself does not specify the multiprocessing interface?

                launcher = self
                # the consumer, which is of type ProcessTask, takes the two queues, and a newly
                # created launcher object. See comments in ProcessTask for more details
                t = ProcessTask(tasks_queue, gpu_allocation_queue, launcher) #,return_queue
                t.set_dependencies(dependencies, lock)
                # start the consumers, which are now waiting for tasks to appear in the task queue.
                # Notice that at the moment tasks_queue is empty
                t.start()

        # If parallelism is 2, then I need to create multiple queues for GPUs, one per each node
        # from "nodes" in the conf file. Notice that the tasks_queue is unique
        if parallelism==2:
            tasks_queue = multiprocessing.JoinableQueue()
            gpu_allocation_queue = []

            poolSizes = []
            gpu_allocation_queues = []

            for node in config["nodes"]:
                p = len(node["used_GPUs"])*node["cores_per_GPU"]
                poolSizes.append(p)

                q = multiprocessing.JoinableQueue()
                for i in range(node["cores_per_GPU"]):
                    for j in node["used_GPUs"]:
                        q.put(j)
                gpu_allocation_queues.append(q)

            for i, node in enumerate(config["nodes"]):
                for t in range(poolSizes[i]):
                    # launcher = self.__class__(self.launchableClass, dataset)

                    launcher = self

                    t = ProcessDistributedTask(tasks_queue, gpu_allocation_queues[i], launcher, i, node["IP"]) #,return_queue
                    t.start()

            # this is to count how many poison pills I will need
            poolSize = sum(poolSizes)

        # Now that consumers have been creates, I take care of the tasks, by adding them tasks_queue,
        # so taht the consumer can start to do their job

        # By calling create_opts_list, I create the list of tasks by computing the Cartesian product
        # between the lists of the entries in the model_params dictionary

        # Notice that this function can be overwritten by a child Launcher, in case the algorithm
        # needs other algorithm to work. In this case in the conf file, model_params has a reference to
        # other conf files, and thus each element in the opts_list could be a tuple and not a
        # single element. Refer to TestAdvExamplesLauncher for an example of such behavior.
        # The returned_config in this case becomes a tuple of config files, which includes the config
        # of the algorithms used by the main algorithm to which model_params and config refer.
        dmopts_list, returned_config = self.create_opts_list(dataset_params, model_params, config)

        # for each task in the list, either add it to the tasks_queue, or excecuted it now if
        # parallism is 0
        for task in dmopts_list:
            # if I am running with queues, I add the task to the unique tasks_queue (for both pool
            # or distributed)
            if parallelism == 1 or parallelism == 2:
                # notice that if I don't copy the task (which is a dictionary) the iterator changes
                # the task at each iteration, and this had consequences in changing also the tasks
                # already added in the queue. This is because everything is a reference in Python
                tasks_queue.put((copy.deepcopy(task), returned_config))
            # otherwise I execute directly the task with no Processes
            elif parallelism == 0:
                # #TODO-ARGO2 self.__class__(...) this is very bad practice... :( can it be avoided?
                # # notice that also here I create a new object of type self
                # launcher = self.__class__(self.launchableClass, dataset)
                launcher = self
                # notice that I choose the first available GPU from nodes[0]
                launcher.task_execute(task, returned_config, gpu=list(config["nodes"][0]["used_GPUs"])[0])

        # now, there is one last task for each comsumer to be done
        # all of them they have done their job, and they are disposable, so we can give them
        # "the pill". They will know what to do.. (harakiri)
        if parallelism==1 or parallelism==2:
            # here we need to wait for the queue to be empty, becausa tasks may be queued as they are
            # processed
            for task in dmopts_list:
                tasks_queue.join()
                print("Task done")

            # at his point, all task have been completed, thus I can kill the processes

            # poison pill for each consumer
            for i in range(poolSize):
                tasks_queue.put(None)
                # I wait for somebody to execute the order
                # Notice that I don't know will do it first, what I need is to join a number of times
                # equal to poolSize
                tasks_queue.join()

        print("Everything is done")

        '''
        # kill all threads in each queue, since I have a queue for each computing node
        if parallelism==2:
            # poison pill for each consumer
            for i, node in enumerate(config["nodes"]):
                for i in range(poolSizes[i]):
                    tasks_queue.put(None)
                    tasks_queue.join()
        '''

        '''
        elif network==1:
            # server

            # create server architecture

            # lanch clients on ssh
            for ip in config["nodes"]:
                prin

        elif network==2:
            # client
        else:
            raise Exception("Network option not recognized")
        '''

# TODO not working currently need to be refactored to handle dataset_params
    # def task_execute_remote(self, opts, config, gpu=-1, ip="127.0.0.1", message_prefix=""):
    #     # TODO this willbe fixed when the optimization module will be created
    #     assert(self._dataset is not None), "dataset has not been set"
    #
    #     if message_prefix != "":
    #         message_prefix += " "
    #
    #     fileName = self.get_algorithm_id(opts)
    #
    #     print(message_prefix + "got a new job, checking " + fileName)
    #
    #     path_log = config["dirName"] + '/' + fileName + '.log'
    #     if not os.path.isfile(path_log):
    #         print("Consumer " + str(os.getpid()) + " " + fileName)
    #
    #         dirName = config["dirName"]
    #         path_conf = dirName + '/' + fileName + '.conf'
    #
    #         f = open(path_conf, 'w')
    #         f.write(str(self._dataset._params_original))
    #         f.write(",")
    #         f.write(str(opts))
    #         f.write(",")
    #         f.write(str(config))
    #         f.close()
    #
    #         cmd = ["ssh", ip, "cd", os.getcwd(), "&&", "python3", sys.argv[0], path_conf, "single"]
    #         print(message_prefix + "executing: " + ' '.join(cmd))
    #         # this is not returning realtime output
    #         subprocess.call(cmd)
    #         #sub_process = subprocess.Popen(cmd, close_fds=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #
    #         #from http://blog.kagesenshi.org/2008/02/teeing-python-subprocesspopen-output.html
    #         #proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1)
    #
    #         #while True:
    #         #    line = proc.stdout.readline()
    #         #    #stdout.append(line)
    #         #    #print(line.decode('utf-8'), end='', flush=True)
    #         #    print(line.decode('utf-8'))
    #         #    if line.decode('utf-8') == '' and proc.poll() != None:
    #         #        break
    #
    #         #while subprocess.poll() is None:
    #         #    for c in iter(lambda: subprocess.stdout.read(1) if subprocess.poll() is None else {}, b''):
    #         #        c = c.decode('ascii')
    #         #        sys.stdout.write(c)
    #         #sys.stdout.flush()
    #
    #         #stdout = []
    #         #while True:
    #         #    line = p.stdout.readline()
    #         #    #stdout.append(line)
    #         #    print("Line: " + line)
    #         #    if line == '' and p.poll() != None:
    #         #        break
    #
    #         #print(message_prefix + "sent job " + fileName)
    #     else:
    #         print(message_prefix + "found completed job " + fileName)
