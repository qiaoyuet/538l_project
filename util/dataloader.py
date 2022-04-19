import os

import pickle
import json
import torch
import haiku as hk

import jax.tree_util
import numpy as np
import jax.numpy as jnp
from torch.utils import data

'''
Custom dataloader so that we can use Pytorch dataloaders and datasets with Jax.
'''


class NumpyLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             sampler=sampler,
                                             batch_sampler=batch_sampler,
                                             num_workers=num_workers,
                                             collate_fn=numpy_collate,
                                             pin_memory=pin_memory,
                                             drop_last=drop_last,
                                             timeout=timeout,
                                             worker_init_fn=worker_init_fn)


'''
Cast values to float32. Used in transformations for dataloader.
'''


class Cast(object):
    def __call__(self, pic):
        return np.array(pic, dtype=np.float32)


'''
Normalize transformation used by dataloader.
'''


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, pic):
        return (pic / 255. - self.mean) / self.std


class AddChannelDim(object):
    def __call__(self, pic):
        return np.expand_dims(pic, axis=-1)


'''
Collate function that transforms tensors into numpy arrays
'''


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
