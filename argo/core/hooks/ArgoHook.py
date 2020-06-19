import matplotlib
import tensorflow as tf

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pdb

import numpy as np

from datasets.Dataset import short_name_dataset, linestyle_dataset, check_dataset_keys_not_loop

from ..utils.argo_utils import compose_name, create_list_colors

import os

EPOCHS = "epochs"
STEPS = "steps"

from ..argoLogging import get_logger

tf_logging = get_logger()


class ArgoHook(tf.train.SessionRunHook):


    def __init__(self, model,
                 period,
                 time_reference,
                 datasets_keys,
                 plot_offset,
                 tensorboard_dir=None,
                 trigger_summaries=False,
                 extra_feed_dict={}):

        time_choices = [EPOCHS, STEPS]
        if not time_reference in time_choices:
            raise ValueError("time reference attribute can be only in %s" % time_choices)

        self._timer = tf.train.SecondOrStepTimer(every_secs=None,
                                                 every_steps=period)

        self._time_reference_str = time_reference
        self._time_ref_shortstr = self._time_reference_str[:2]

        self._model = model

        self._plot_offset = plot_offset
        self._extra_feed_dict = extra_feed_dict
        # called in before_run
        self._nodes_to_be_computed_by_run = {}

        check_dataset_keys_not_loop(datasets_keys)
        self._datasets_keys = datasets_keys
        self._ds_initializers = model.datasets_initializers
        self._ds_handles_nodes = model.datasets_handles_nodes
        self._ds_handle = model.ds_handle

        # these needs to be defined in the child class
        self._tensors_names = None
        self._tensors_plots = None
        self._tensors_values = None

        self._trigger_summaries = trigger_summaries
        self._tensorboard_dir = tensorboard_dir
        assert not self._trigger_summaries or self._tensorboard_dir is not None, \
            "If you specified that you want to Trigger Summeries, you should also specify where to save them."
        self.SUMMARIES_KEY = "log_mean_summaries"

        self._default_plot_bool = True

    def _set_summaries_filewriters(self):
        """This function sets summaryFileWriters for the local summaries
        it needs to be invoked before training to keep track of the summaries.
        (better invoke it as late as possible, since the FileWriter will corrupt data in the logfolder at each initialization)
        """
        # NB do NOT use FileWriterCache, because it will also write the graph in the summary and it will thus do it every time the model is reloaded
        if self._trigger_summaries:
            self.summary_writers = tf.summary.FileWriter(self._tensorboard_dir)

    def _register_summary_for_tensor(self, name, mn):
        if self._trigger_summaries:
            tf.summary.scalar(name, mn, collections=[self.SUMMARIES_KEY])
        self.summary_nodes = tf.get_collection(self.SUMMARIES_KEY)

    def _write_summaries(self, run_context):
        if self._trigger_summaries:
            summaries = run_context.session.run(self.summary_nodes)
            for summ in summaries:
                self.summary_writers.add_summary(summ, self._global_step)

    # log to self._files
    def log_to_file_and_screen(self, log_to_screen=False):

        firstLog = True
        for i, (tensors_vertical_panel, files_panel) in enumerate(zip(self._tensors_names,
                                                                      self._files)):
            if len(tensors_vertical_panel) > 0:

                if firstLog:
                    time_ref_shortstr = self._time_reference_str[0]
                    logstring = "[" + time_ref_shortstr + " " + str(self._time_ref) + "]"
                else:
                    logstring = ""
                # here it start the vertical panel
                for j, (tensors_names_panel, file_plot) in enumerate(zip(tensors_vertical_panel, files_panel)):
                    # log to file
                    line = str(self._time_ref)

                    for dataset_str in self._datasets_keys:
                        logstring += " ".join(
                            [" " + compose_name(name, short_name_dataset[dataset_str]) + " " + "%.4g" % mean
                             for (name, mean) in zip(tensors_names_panel, self._tensors_values[dataset_str][i][j])])
                        line += "\t" + "\t".join(["%.5g" % mean for mean in self._tensors_values[dataset_str][i][j]])

                    line += "\t" + "%.2f" % self._elapsed_secs

                    line += "\n"
                    self._log_to_file(line, file_plot)

                logstring += "  (%.2fs)" % self._elapsed_secs

                # log to screen
                if firstLog and log_to_screen:
                    tf_logging.info(logstring)
                    firstLog = False

    def before_run(self, run_context):
        self._before_run_operations()

        args = self._before_run_args()

        return tf.train.SessionRunArgs(args)

    def _before_run_args(self):  # , run_context):
        args = {
            "globals": (self._global_step_tensor, self._time_reference_node),
            **self._nodes_to_be_computed_by_run}

        return args

    def _before_run_operations(self):
        self._trigged_for_step = (
                self._next_step is not None and self._timer.should_trigger_for_step(self._next_step))

    def after_run(self, run_context, run_values):

        self._global_step, time_ref = run_values.results["globals"]
        self._time_ref = self.cast_time_ref(time_ref)

        self._next_step = self._global_step + 1

        if self._trigged_for_step:
            # write summaries
            self._write_summaries(run_context)

    def after_create_session(self, session, coord):
        self._next_step = None
        global_step = session.run(self._global_step_tensor)

        n_past_triggers = global_step//self._timer._every_steps
        if global_step%self._timer._every_steps==0 and global_step!=0:
            n_past_triggers-=1

        last_trig_step =  n_past_triggers * self._timer._every_steps
        self._timer.update_last_triggered_step(last_trig_step)

        # this is needed when evaluting session run and passing the handle
        self._ds_handles = session.run(self._ds_handles_nodes)

    # update time
    def update_time(self):
        self._elapsed_secs, self._elapsed_steps = self._timer.update_last_triggered_step(self._global_step)

    def before_training(self, session):
        self._reset_file(session)

    def _log_to_file(self, line, file_panel):
        if file_panel:
            file_panel.write(line)
            file_panel.flush()

    def _create_or_open_files(self):
        self._filesExist = []
        self._files = []
        for (tensors_vertical_panels, tensors_name_vertical_panels) in zip(self._tensors_names,
                                                                           self._tensors_plots):
            filesExist_panel = []
            files_panel = []
            for tensors_names_panel, tensors_plots_panel in zip(tensors_vertical_panels, tensors_name_vertical_panels):

                # create file handler and open file
                filePath = self._dirName + '/' + tensors_plots_panel["fileName"] + '.txt'
                filesExist_panel.append(os.path.isfile(filePath))

                if not filesExist_panel[-1]:
                    files_panel.append(open(filePath, 'w'))
                    self._write_log_header(files_panel[-1], tensors_names_panel)
                else:
                    files_panel.append(open(filePath, 'r+'))

            self._filesExist.append(filesExist_panel)
            self._files.append(files_panel)

    def _write_log_header(self, file_panel, tensors_names_panel):
        header = self._time_reference_str + "\t"
        for dataset_str in self._datasets_keys:
            header += " ".join([" " + compose_name(name, dataset_str) + "\t " for name in tensors_names_panel])
        header += "time_sec"
        self._log_to_file("# " + header + "\n", file_panel)

    def cast_time_ref(self, time_ref):
        # for printing purposes if epoch period turns out to be an integer then I don't want to print the zeros, e.g. 1.0
        if isinstance(time_ref, (np.floating, float)) and np.mod(time_ref, 1) == 0:
            time_ref = int(time_ref)

        return time_ref

    def _reset_file(self, session):
        # reset log file
        # I need to change the handle in the iterator, so I cannot do a for ... in ...
        # this is not pythonic, sorry for that (Luigi 19/03/19)
        for i in range(len(self._files)):
            for j in range(len(self._files[i])):

                if self._files[i][j] and self._filesExist[i][j]:

                    # remove extra lines from the log file
                    time_ref = session.run(self._time_reference_node)
                    self._time_ref = self.cast_time_ref(time_ref)

                    lines = self._files[i][j].readlines()
                    c = 0
                    for l, line in enumerate(lines[:]):
                        splited = line.replace('\x00','').split("\t")
                        if not line.startswith("#") and self._cast(splited[0]) >= self._time_ref:
                            break
                        c = c + 1

                    fileName = self._files[i][j].name

                    self._files[i][j].close()
                    self._files[i][j] = open(fileName, 'w')
                    self._files[i][j].writelines(lines[:c])
                    self._files[i][j].flush()

    def begin(self):
        # from doc: "Second call of begin() on the same graph, should not change the graph."
        # https://www.tensorflow.org/api_docs/python/tf/train/SessionRunHook

        if not self._did_begin_already:
            self._global_step_tensor = tf.train.get_global_step()
            self._global_epoch = tf.get_collection("global_epoch")[0]

            if self._time_reference_str == EPOCHS:
                self._time_reference_node = self._global_epoch
                self._cast = float
            elif self._time_reference_str == STEPS:
                self._time_reference_node = self._global_step_tensor
                self._cast = int

        self._set_summaries_filewriters()

    def end(self, session):
        # close files
        for vertical_panels in self._files:
            for file_panel in vertical_panels:
                if file_panel:
                    file_panel.close()

    def plot(self):

        if not self._default_plot_bool:
            return

        n_columns = len(self._tensors_plots)
        n_rows = np.max([len(p) for p in self._tensors_plots])

        # create figure
        fig = plt.figure(figsize=(20, 9))
        fig.suptitle(self._model.id, y=0.995, fontsize=10)

        c = 0

        for (files_vertical_panels, tensors_names_vertical_panels, tensors_plots_vertical_panels) in zip(self._files,
                                                                                                         self._tensors_names,
                                                                                                         self._tensors_plots):

            r = 0

            for (file_panel, tensors_names_panel, tensors_plot_panel) in zip(files_vertical_panels,
                                                                             tensors_names_vertical_panels,
                                                                             tensors_plots_vertical_panels):


                # read data from file
                with open(file_panel.name) as f:
                    data = f.read()
                    
                split_data = data.split('\n')

                # remove lines starting with #
                first_line = split_data[0]
                while first_line[0] == "#":
                    split_data = split_data[1:]
                    first_line = split_data[0]


                # this is not ok, we should check for the value, not how many I skip
                data = split_data[self._plot_offset:-1]


                x = [self._cast(row.replace('\x00','').split("\t")[0]) for row in data]

                n_plots = len(tensors_names_panel)

                ax = fig.add_subplot(n_rows, n_columns, r * n_columns + c + 1)

                max_colors = len(tensors_names_panel) # see line below to check only 10 colors are generated
                list_colors = create_list_colors(max_colors)

                for color, name in enumerate(tensors_names_panel):

                    for j, dataset_str in enumerate(self._datasets_keys):

                        if tensors_plot_panel.get("compose-label", 1):
                            label = compose_name(name, dataset_str, separator=" ")
                        else:
                            label = name

                        y = [float(row.split("\t")[color + 1 + j * n_plots]) for row in data]

                        if tensors_plot_panel.get("logscale-y", 0):
                            # check for error bars
                            if tensors_plot_panel.get("error-bars", 0) and color % 2 == 1:
                                pdb.set_trace()
                                ax.errorbar(x,
                                            y_curve,
                                            y,
                                            fmt='None',
                                            c=list_colors[color % len(list_colors)],
                                            #ms=20,
                                            #mew=4
                                            )
                                y_curve = y
                                ax.set_yscale('log')
                            else:
                                ax.semilogy(x,
                                            y,
                                            linestyle_dataset[dataset_str],
                                            c=list_colors[color-1 % len(list_colors)],
                                            label=label)
                                y_curve = y
                        else:
                            # check for error bars
                            if tensors_plot_panel.get("error-bars", 0) and color % 2 == 1:
                                ax.errorbar(x,
                                            y_curve,
                                            y,
                                            fmt='',
                                            linestyle = 'None',
                                            c=list_colors[color-1 % len(list_colors)],
                                            #ms=20,
                                            #mew=4
                                            )
                            else:
                                ax.plot(x,
                                        y,
                                        linestyle_dataset[dataset_str],
                                        c=list_colors[color % len(list_colors)],
                                        label=label)
                                y_curve = y



                ax.set_xlabel(self._time_reference_str)
                ax.set_ylabel(tensors_plot_panel["fileName"])
                ax.set_xlim(left=self._plot_offset)

                if tensors_plot_panel.get("legend", 1):
                    ax.legend()

                # ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
                # ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))

                xt = ax.get_xticks()
                xt = np.append(xt, self._plot_offset)
                xtl = xt.astype(int).tolist()
                xtl[-1] = self._plot_offset
                ax.set_xticks(xt)
                ax.set_xticklabels(xtl)

                ax.grid()

                r += 1

            c += 1

        plt.tight_layout()

        plt.savefig(self._dirName + "/" + self._fileName + ".png")
        plt.close()
