import sonnet as snt
from .AbstractModule import AbstractModule


class BatchNorm(AbstractModule):
    def __init__(self, is_training, name="BatchNorm", **kwargs):
        super().__init__(name=name)
        self._is_training = is_training
        self._name = name
        self._kwargs = kwargs

    def _build(self, inputs):
        bn = snt.BatchNorm(**self._kwargs)
        return bn(inputs, is_training=self._is_training, test_local_stats=False)


# # the purpose of this class is to set is_training = True in build()
# class BatchNorm(snt.BatchNorm):
#
#     def _build(self, input_batch):
#         return super(BatchNorm, self)._build(input_batch, is_training=True, test_local_stats=True)
