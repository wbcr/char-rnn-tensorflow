# -*- coding: utf-8 -*-

import codecs
import collections
import numpy as np
import os
import random
from six.moves import cPickle


class TextLoaderBase():
    """Base class with common methods."""

    DATA_ITEMS=[]

    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, 'input.txt')
        vocab_file = os.path.join(data_dir, 'vocab.pkl')
        data_file = os.path.join(data_dir, 'data.npz')

        if not (os.path.exists(vocab_file) and os.path.exists(data_file)):
            print('reading text file')
            self.preprocess(input_file)
            self.save_preprocessed(vocab_file, data_file)
            self.postprocess()
        else:
            print('loading preprocessed files')
            self.load_preprocessed(vocab_file, data_file)
            self.postprocess()

        self.create_batches()
        self.reset_batch_pointer()

    def load_preprocessed(self, vocab_file, data_file):
        self.load_vocab(vocab_file)
        with np.load(data_file) as data:
            for name in data.files:
                setattr(self, name, data[name])

    def save_preprocessed(self, vocab_file, data_file):
        self.save_vocab(vocab_file)
        kwargs = {name: getattr(self, name) for name in self.DATA_ITEMS}
        np.savez(data_file, **kwargs)

    def create_vocab(self, data):
        self.chars = tuple([char for char, count in collections.Counter(data).most_common()])
        self.vocab = {c: i for i, c in enumerate(self.chars)}

    def load_vocab(self, vocab_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab = {c: i for i, c in enumerate(self.chars)}

    def save_vocab(self, vocab_file):
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)

    def chars_to_array(self, data):
        # This is not very efficient, but we can't use a generator,
        # because numpy needs to know the size of the array in advance.
        return np.array([self.vocab[c] for c in data])

    def create_batches(self):
        self.num_batches = int(self.xtensor.size / (self.batch_size * self.seq_length))
        # When the data (tensor) is too small,
        # let's give them a better error message
        if self.num_batches == 0:
            assert False, 'Not enough data. Make seq_length and batch_size small.'

        # Drop remainder
        self.xtensor = self.xtensor[:self.num_batches * self.batch_size * self.seq_length]
        self.ytensor = self.ytensor[:self.num_batches * self.batch_size * self.seq_length]

        # Split batches
        self.x_batches = np.split(
            self.xtensor.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(
            self.ytensor.reshape(self.batch_size, -1), self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def vocab_size(self):
        return len(self.chars)

    def reset_batch_pointer(self):
        self.pointer = 0

    def preprocess(self):
        pass

    def postprocess(self):
        pass


class TextPredictionTaskLoader(TextLoaderBase):
    """In this task, the RNN has to predict the following char."""

    DATA_ITEMS=['tensor']

    def preprocess(self, input_file):
        with codecs.open(input_file, 'r', encoding=self.encoding) as f:
            data = f.read()
        self.create_vocab(data)
        self.tensor = self.chars_to_array(data)

    def postprocess(self):
        # Shift characters by one to learn to predict.
        self.xtensor = self.tensor
        self.ytensor = np.copy(self.tensor)
        self.ytensor[:-1] = self.xtensor[1:]
        self.ytensor[-1] = self.xtensor[0]


class TextCorrectionTaskLoader(TextLoaderBase):
    """In this task, the RNN has to replace a char."""

    DATA_ITEMS=['xtensor', 'ytensor']

    def preprocess(self, input_file):
        with codecs.open(input_file, 'r', encoding=self.encoding) as f:
            data = f.read()
        self.create_vocab(data)

        self.ytensor = self.chars_to_array(data)
        xdata = self.mapstring(data)
        self.xtensor = self.chars_to_array(xdata)

    def mapstring(self, data):
        # Replace characters with diacritics and y with i.
        intab  = u'yýíáéúóŕžščďřťňľäô'
        outtab = u'iiiaeuorzscdrtnlao'
        transtab = {ord(i): o for i, o in zip(intab, outtab)}

        return data.translate(transtab)
