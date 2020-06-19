import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np

class PandasLogger:

    def __init__(self, logdir, loggername, columns_names, try_to_convert={},
                 field_to_sort_by=None, replace_strings_for_sort={}):
        self._filename = os.path.join(logdir, loggername + ".txt")
        self._plotfilename = os.path.join(logdir, loggername)

        self._pd_csv_kwargs = {
            "sep" : " "
        }

        read_success, _df = self.try_read_csv()
        if read_success:
            self._df = _df
        else:
            self._df = pd.DataFrame(columns=columns_names)

        for field, try_conv in try_to_convert.items():
            self._df[field] = [try_conv(x) for x in self._df[field].array]

        self._field_to_sort_by = field_to_sort_by
        self._replace_strings_for_sort = replace_strings_for_sort

        self._maybe_sort_df()

        os.makedirs(logdir, exist_ok=True)

    def _maybe_sort_df(self):
        if self._field_to_sort_by is not None:

            for string, value in self._replace_strings_for_sort.items():
                self._df[self._field_to_sort_by].replace(to_replace=string, value=value, inplace=True)

            self._df.sort_values(by=self._field_to_sort_by, inplace=True)

            for string, value in self._replace_strings_for_sort.items():
                self._df[self._field_to_sort_by].replace(to_replace=value, value=string, inplace=True)

    # def has_value(self, field, value):
    #     return (value in self._df[field])
    #
    def has_values(self, fields, values):
        booleans = True
        for field, value in zip(fields, values):
            booleans = booleans & np.array([self._isclose(x, value) for x in self._df[field]], dtype=bool)
        return  booleans.any()

    def _isclose(self, x, val):
        if isinstance(x, float) and isinstance(val, float):
            return np.isclose(x, val, atol=1e-8)
        else:
            return x == val

    def try_read_csv(self):
        read_success = False
        _df = None
        try:
            if os.path.exists(self._filename) and os.path.getsize(self._filename) > 0:
                _df = pd.read_csv(self._filename, **self._pd_csv_kwargs)
                read_success = True
        except:
            pass

        return read_success, _df

    def log(self, log_values_np):
        new_series = pd.Series(log_values_np, index=self._df.columns)

        #NB as of today there is no `inplace` parameter for pandas dataframes
        self._df = self._df.append(new_series, ignore_index=True)
        # self._df.loc[index] = monitor_tensors_values_np
        # self._df.sort_index(inplace=True)
        self._maybe_sort_df()

        self._df.to_csv(self._filename, index=False, float_format='%.6g', **self._pd_csv_kwargs)

    def plot(self, ylims=None, suffix=None, **kwargs):
        # subplots = False, sharex = True
        axes = self._df.plot(**kwargs)

        if ylims is not None:
            for ax, ylim in zip(axes, ylims):
                ax.set_ylim(ylim)

        self._finalize_plot(suffix=suffix)

    def plot_groupby(self, groupfield, suffix=None, **kwargs):
        ax=plt.gca()
        for label, df in self._df.groupby(groupfield):
            df.plot(**kwargs, ax=ax, label=groupfield+"_{:}".format(label))
        self._finalize_plot(suffix=suffix)

    def plot_errorbar(self, x, y, suffix=None, ylim=None, string_replace={}, **kwargs):
        xdata = []
        ydata = []
        yerr = []

        for label, df in self._df.groupby(x):
            try:
                x_value = string_replace[label]
            except:
                x_value = label

            xdata.append(x_value)
            ydata.append(df.mean()[y])
            yerr.append(df.std()[y])

        xdata, ydata, yerr = zip(*sorted(zip(xdata, ydata, yerr)))
        plt.errorbar(xdata, ydata, yerr=yerr, label=y, **kwargs)
        if ylim is not None:
            plt.ylim(ylim)
        plt.legend()
        self._finalize_plot(suffix=suffix)

    def _finalize_plot(self, suffix=None):
        plt.tight_layout()
        plotfilename = self._plotfilename
        if suffix is not None:
            plotfilename+=suffix
        plt.savefig(plotfilename+".png")
        plt.close()
