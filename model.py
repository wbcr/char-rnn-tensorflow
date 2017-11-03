# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.contrib import legacy_seq2seq

import numpy as np

class ModelBase():
    """Model creation base class."""

    def create_var(self, scope, name, *args):
        with tf.variable_scope(scope, reuse=False):
            return tf.get_variable(name, *args)

    def select_cell_fn(self, name):
        cellfns = { 'rnn': rnn.BasicRNNCell,
                    'gru': rnn.GRUCell,
                    'lstm': rnn.BasicLSTMCell,
                    'nas': rnn.NASCell }
        try:
            return cellfns[name]
        except KeyError:
            raise Exception("model type not supported: {}".format(name))

    def create_var(self, scope, name, *args):
        with tf.variable_scope(scope, reuse=False):
            return tf.get_variable(name, *args)


    def create_cell_stack(self, scope, cell_fn, args, use_dropout=False):
        cells = []
        for layer in range(args.num_layers):
            with tf.variable_scope('%s_%x' % (scope, layer)):
                cell = cell_fn(args.rnn_size)
                if use_dropout:
                    cell = rnn.DropoutWrapper(
                        cell, input_keep_prob=args.input_keep_prob,
                        output_keep_prob=args.output_keep_prob)
                cells.append(cell)
        return cells


class ModelForwardRNN(ModelBase):
    def __init__(self, args, training=True):
        self.args = args
        if not training:
            args.batch_size = 1
            args.seq_length = 1

        use_dropout = training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0)
        cell_fn = self.select_cell_fn(args.model)
        cells = self.create_cell_stack('hidden', cell_fn, args, use_dropout=use_dropout)
        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

        self.input_data = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        softmax_w = self.create_var('rnnlm', 'softmax_w', [args.rnn_size, args.vocab_size])
        softmax_b = self.create_var('rnnlm', 'softmax_b', [args.vocab_size])

        embedding = self.create_var('rnnlm', 'embedding', [args.vocab_size, args.rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # dropout beta testing: double check which one should affect next line
        if training and args.output_keep_prob:
            inputs = tf.nn.dropout(inputs, args.output_keep_prob)

        inputs = tf.split(inputs, args.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])


        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])])
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # instrument tensorboard
        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost)

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else:  # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret


def ModelBidirectionalRNN(ModelBase):

      def __init__(self, args, training=True):

        self.args = args
        if not training:
            args.batch_size = 1

        use_dropout = training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0)

        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])

        embedding = self.create_var('input', 'embedding', [args.vocab_size, args.rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        if use_dropout:
            inputs = tf.nn.dropout(inputs, args.output_keep_prob)

        cell_fn = self.select_cell_fn(args.model)
        cells_fw = self.create_cell_stack('hidden_fw', cell_fn, args, use_dropout=use_dropout)
        cells_bw = self.create_cell_stack('hidden_bw', cell_fn, args, use_dropout=use_dropout)

        self.cell_fw = rnn.MultiRNNCell(cells_fw, state_is_tuple=True)
        self.cell_bw = rnn.MultiRNNCell(cells_bw, state_is_tuple=True)

        self.initial_state = (self.cell_fw.zero_state(args.batch_size, tf.float32),
                              self.cell_bw.zero_state(args.batch_size, tf.float32))

        sequence_length = [args.seq_length]*args.batch_size
        outputs, self.final_state = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, inputs, sequence_length,
            initial_state_fw=self.initial_state[0], initial_state_bw=self.initial_state[1])

        # bidi dynamic rnn does not concatenate fw and bw cells by default
        output = tf.concat(outputs, 2, name="concat_outputs")

        softmax_w = self.create_var('rnlm', 'softmax_w', [2*args.rnn_size, args.vocab_size])
        softmax_b = self.create_var('rnlm', 'softmax_b', [args.vocab_size])

        # Reshape/matmul/reshape sequence
        self.logits = tf.einsum("ijk,kl->ijl", output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)

        loss = seq2seq.sequence_loss(self.logits, self.targets, tf.ones([args.batch_size, args.seq_length]), average_across_batch=False)

        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length

        self.lr = tf.Variable(0.0, trainable=False)

        # apply clipping
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # instrument tensorboard
        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost)
