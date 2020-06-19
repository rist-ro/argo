from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
from tqdm import tqdm

class ConfidenceIntervalsOnlySamples:

    def __init__(self,
                 dirName,
                 dataset_initializers,
                 dataset_handles,
                 handle,
                 n_samples_ph,
                 distsample,
                 raw_x,
                 x,
                 y,
                 posterior_samples=2500,
                 n_batches=-1,
                 extra_nodes_to_collect = [],
                 extra_nodes_names = []
                ):

        super().__init__()
        #        self._datasets_keys= datasets_keys
        self._posterior_samples = posterior_samples
        #        self._session=session
        self._dirName = dirName

        self._distsample = distsample

        self._raw_x=raw_x
        self._y=y
        self._x = x
        self._dataset_initializers=dataset_initializers
        self._dataset_handles=dataset_handles
        self._handle=handle
        self._n_batches=n_batches
        self._n_samples_ph = n_samples_ph

        if len(extra_nodes_to_collect)!=len(extra_nodes_names):
            raise ValueError("extra_nodes_names should have same length than "
                             "extra_nodes_to_collect, found: `{:d}` and `{:d}`".format(len(extra_nodes_to_collect), len(extra_nodes_names)))

        self._extra_nodes_to_collect = extra_nodes_to_collect
        self._extra_nodes_names = extra_nodes_names


    def do_when_triggered(self, session, datasets_keys, timeref, timeref_str="ep"):
        for ds_key in datasets_keys:
            self._calculate_mc_dropout(session, ds_key, timeref, timeref_str)

    def _create_name(self, prefix, baseName):
       return self._dirName + "/" + prefix + "-" + baseName

    def _calculate_mc_dropout(self, session, ds_key, timeref, timeref_str):

        if type(session).__name__ != 'Session':
            raise Exception("I need a raw session to evaluate metric over dataset.")

        dataset_initializer = self._dataset_initializers[ds_key]
        dataset_handle = self._dataset_handles[ds_key]
        baseName=ds_key+"-{:}{:04d}".format(timeref_str, timeref)

        init_ops = dataset_initializer
        session.run(init_ops)
        batch_samples_list = []
        real_valu_list = []

        extra_batch_dict = {}
        for name in self._extra_nodes_names:
            extra_batch_dict[name] = []

        pbar = tqdm(desc='collecting samples {:}'.format(ds_key),
                    total=(self._n_batches if (self._n_batches != -1) else None), dynamic_ncols=True)
        b = 0

        while True:

            batch_extras = {}
            for name in self._extra_nodes_names:
                batch_extras[name] = []

            batch_samples = []

            try:
                # model.raw_x is the input before any noise addition (if present), we want to make sure we get the clean batch before noise
                batch_x, batch_y = session.run([self._raw_x, self._y],
                                               feed_dict={self._handle: dataset_handle})

                # model.x is the input after noise addition (if present), we want to make sure we feed x, so that noiose will not be added.
                for mcm in range(self._posterior_samples):
                    samples, extra_np = session.run([self._distsample, self._extra_nodes_to_collect],
                                          feed_dict={self._x: batch_x, self._n_samples_ph: 1})

                    batch_samples.append(samples)

                    for name, arr in zip(self._extra_nodes_names,extra_np):
                        batch_extras[name].append(arr)


                batch_samples_stack = np.stack(batch_samples, axis=2)

                batch_samples_list.append(batch_samples_stack)
                real_valu_list.append(batch_y)

                for name in self._extra_nodes_names:
                    extra_batch_dict[name].append(
                                            np.stack(batch_extras[name], axis=2)
                    )

                b += 1
                pbar.update(1)

                if b == self._n_batches:
                    break

            except tf.errors.OutOfRangeError:
                break


        np.save(self._create_name('batch_reals', baseName), real_valu_list)
        np.save(self._create_name('batch_samples', baseName), batch_samples_list)

        for name in self._extra_nodes_names:
            np.save(self._create_name(name, baseName), extra_batch_dict[name])

        self._stats_and_plot(baseName, batch_samples_list, real_valu_list, extra_batch_dict)


    @abstractmethod
    def _stats_and_plot(self, baseName, batch_samples_list, real_valu_list, extra_batch_dict):
        pass

