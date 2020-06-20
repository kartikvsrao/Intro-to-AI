# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

import os
from collections import namedtuple
import gzip
import numpy as np

__all__ = [
    "load_csv", "load_mnist"
]

CURRENT_DIR = os.path.dirname(__file__)

class Dataset(object):

    def __init__(self, data, labels, batch_size=None, n_classes=None, drop_remainder=None, random=True):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self._n_classes = n_classes
        self.cursor = 0
        self._idx = np.arange(self.n_examples)
        self.random = random
    
    @property
    def n_examples(self):
        return len(self.data)
    
    @property
    def n_classes(self):
        return np.max(self.labels)+1 if self._n_classes is None else self._n_classes

    def next_batch(self, batch_size=None, drop_remainder=None, _is_iter=False):
        if batch_size is None:
            batch_size = self.batch_size
        if batch_size is None:
            print("[Error] Batch size for the database is not defined")
            if _is_iter:
                raise StopIteration
            else:
                return
        if drop_remainder is None:
            drop_remainder = self.drop_remainder
        if self.cursor == 0 or self.cursor >= self.n_examples or (drop_remainder and self.cursor+batch_size > self.n_examples):
            if self.cursor != 0 and _is_iter:
                raise StopIteration
            self.cursor = batch_size
            if self.random: np.random.shuffle(self._idx)
            self.data = self.data[self._idx]
            self.labels = self.labels[self._idx]
        else:
            self.cursor += batch_size
        return self.data[self.cursor-batch_size:self.cursor], self.labels[self.cursor-batch_size:self.cursor]

    def __iter__(self):
        self.cursor = 0
        return self

    def __next__(self):
        return self.next_batch(_is_iter=True)


def load_csv(f, header=1):
    d = np.loadtxt(os.path.join(CURRENT_DIR, f), delimiter=',', skiprows=header)
    return d[:, :-1], d[:, -1]


def load_mnist(target_dir=os.path.join(CURRENT_DIR, "mnist"), batch_size=None, validation_size=0.1):
    train_data = 'train-images-idx3-ubyte.gz'
    train_labels = 'train-labels-idx1-ubyte.gz'
    test_data = 't10k-images-idx3-ubyte.gz'
    test_labels = 't10k-labels-idx1-ubyte.gz'

    # data loader from tensorflow
    # https://github.com/tensorflow/tensorflow/blob/7c36309c37b04843030664cdc64aca2bb7d6ecaa/tensorflow/contrib/learn/python/learn/datasets/mnist.py#L189
    def _read32(bytestream):
        dt = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(bytestream.read(4), dtype=dt)[0]
    def extract_data(f):
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = _read32(bytestream)
            if magic != 2051:
                raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                                (magic, f.name))
            num_images = _read32(bytestream)
            rows = _read32(bytestream)
            cols = _read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, 1, rows, cols)
            return data
    def extract_labels(f, one_hot=False):
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = _read32(bytestream)
            if magic != 2049:
                raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                                (magic, f.name))
            num_items = _read32(bytestream)
            buf = bytestream.read(num_items)
            labels = np.frombuffer(buf, dtype=np.uint8)
            return labels

    with open(os.path.join(target_dir, train_data), "rb") as f:
        train_data = extract_data(f)
    with open(os.path.join(target_dir, train_labels), "rb") as f:
        train_labels = extract_labels(f)
    with open(os.path.join(target_dir, test_data), "rb") as f:
        test_data = extract_data(f)
    with open(os.path.join(target_dir, test_labels), "rb") as f:
        test_labels = extract_labels(f)
    if validation_size > 1:
        validation_size = int(validation_size)
    else:
        validation_size = int(validation_size*len(train_data))
    val_data = train_data[:validation_size]
    val_labels = train_labels[:validation_size]
    train_data = train_data[validation_size:]
    train_labels = train_labels[validation_size:]
    
    return namedtuple("Dataset", ["train", "validation", "test", "n_classes", "feature_shape"])(
        Dataset(train_data, train_labels, batch_size=batch_size, n_classes=10, drop_remainder=False, random=True),
        Dataset(val_data, val_labels, batch_size=batch_size, n_classes=10, drop_remainder=False, random=False),
        Dataset(test_data, test_labels, batch_size=batch_size, n_classes=10, drop_remainder=False, random=False),
        10, train_data.shape[1:]
    )
