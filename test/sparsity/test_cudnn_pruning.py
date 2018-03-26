import tensorflow as tf
import unittest

from util.sparsity import cudnn_pruning

class CudnnLSTMTest(unittest.TestCase):

    def setUp(self):
        self.batch_size = 8
        self.input_size = 2
        self.time = 5
        self.units = 10
        self.input_shape = [self.time, self.batch_size, self.input_size]

    def testMaskedCudnnLSTM(self):
        expected_num_masks = 8

        lstm_cell = cudnn_pruning.MaskedCudnnLSTM(num_layers=1,
                                                  num_units=self.units)
        lstm_cell.build(self.input_shape)

        inputs = tf.Variable(tf.random_normal(self.input_shape))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(lstm_cell(inputs))
