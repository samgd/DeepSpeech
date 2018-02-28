import numpy as np
import tensorflow as tf
import unittest

from util import opaque_params

class TestOpaqueParams(unittest.TestCase):

    def setUp(self):
        self.n_units = 2048
        self.n_input = 4096
        self.num_dirs = 2
        self.opaque_size = self.num_dirs * (4*self.n_input*self.n_units +
                                            4*self.n_units*self.n_units +
                                            8*self.n_units)
        self.params = np.random.randn(self.opaque_size).astype(np.float32)

    def test_identity(self):
        var_names_to_values = opaque_params.split(self.params)
        joined_params = opaque_params.join(var_names_to_values)
        self.assertEqual(joined_params.shape, self.params.shape)
        self.assertTrue(np.all(joined_params == self.params))

    def test_opaque_params_split(self):
        var_names_to_values = opaque_params.split(self.params)

        # Create TensorFlow graph and converter.
        opaque_var = tf.get_variable('opaque_var', initializer=self.params)
        converter = tf.contrib.cudnn_rnn.CudnnLSTMSaveable(opaque_var,
                                                           num_layers=1,
                                                           num_units=self.n_units,
                                                           input_size=self.n_input,
                                                           direction='bidirectional')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            weights, biases = converter._OpaqueParamsToCanonical()

            tf_vals = {}
            # Forward.
            tf_vals.update(cudnn_to_weights(weights[:8], prefix='fw_'))
            tf_vals.update(cudnn_to_biases(biases[:8], prefix='fw_'))
            # Backward.
            tf_vals.update(cudnn_to_weights(weights[8:], prefix='bw_'))
            tf_vals.update(cudnn_to_biases(biases[8:], prefix='bw_'))

            for name, value in var_names_to_values.items():
                tf_val = sess.run(tf_vals[name])
                self.assertEqual(value.shape, tf_val.shape,
                                 msg='np and tf vals have different shapes for %s' % name)
                self.assertTrue(np.allclose(value, tf_val),
                                msg='np and tf vals differ for %s' % name)

def cudnn_to_weights(cu_weights, prefix=None):
    '''Version of _cudnn_to_tf_weights to return a dict of name->Tensor.

    https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/contrib/cudnn_rnn/python/ops/cudnn_rnn_ops.py#L495
    '''

    names = ['Wi', 'Wf', 'Wc', 'Wo',
             'Ri', 'Rf', 'Rc', 'Ro']
    if prefix is not None:
        names = [prefix + name for name in names]
    return dict(zip(names, cu_weights))

def cudnn_to_biases(cu_biases, prefix=None):
    '''Wrapper for _cudnn_to_tf_biases to return a dict of name->Tensor.

    https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/contrib/cudnn_rnn/python/ops/cudnn_rnn_ops.py#L533
    '''
    names = ['b_Wi', 'b_Wf', 'b_Wc', 'b_Wo',
             'b_Ri', 'b_Rf', 'b_Rc', 'b_Ro']
    if prefix is not None:
        names = [prefix + name for name in names]
    return dict(zip(names, cu_biases))
