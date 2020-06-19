from word_embedding.test.core.PandasLogger import PandasLogger
from collections import defaultdict

def instantiate_logger(keytuple, output_folder, log_prefix, names_to_log, try_to_convert={},
                       field_to_sort_by=None, replace_strings_for_sort={}):
    keystring = "_".join(keytuple)
    return PandasLogger(output_folder,
                        "{:}_{:}".format(log_prefix, keystring),
                        names_to_log,
                        try_to_convert=try_to_convert,
                        field_to_sort_by=field_to_sort_by,
                        replace_strings_for_sort=replace_strings_for_sort)


class CustomDefaultDict(defaultdict):
    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value

# loggers = CustomDefaultDict(partial(instantiate_logger,
#                                         output_folder=output_folder,
#                                         log_prefix="{:}_clustering".format(groups_name),
#                                         names_to_log=["purity", "homogeneity", "completeness", "silhouette"]
#                                         )
#                                 )