import copy
import numpy as np
import tensorflow as tf
import unittest

from util import convert_params

class TestConvertParams(unittest.TestCase):

    def setUp(self):
        '''Initialize parameters.'''
        self.n_units = 2048
        self.n_input = 4096
        self.num_dirs = 2

        self.cudnn_params_name = 'fp32_storage/cudnn_lstm/opaque_kernel'
        self.init_cudnn_params()
        self.init_canonical_params()
        self.init_basic_params()
        self.add_non_lstm_values()

    def tearDown(self):
        '''Clear parameter dictionaries to prevent a memory leak.'''
        self.cudnn_params.clear()
        self.canonical_params.clear()
        self.basic_params.clear()

    def init_cudnn_params(self):
        '''Initialize random cudnn_params blob.'''
        self.cudnn_size_weight = (4*self.n_input*self.n_units +
                                  4*self.n_units*self.n_units)
        self.cudnn_size_bias = 8*self.n_units
        self.cudnn_size = self.num_dirs * (self.cudnn_size_weight + self.cudnn_size_bias)

        cudnn_params = np.random.randn(self.cudnn_size)
        adam_params = np.random.randn(self.cudnn_size)
        adam_1_params = np.random.randn(self.cudnn_size)

        self.cudnn_params = {self.cudnn_params_name: cudnn_params,
                             self.cudnn_params_name + '/Adam': adam_params,
                             self.cudnn_params_name + '/Adam_1': adam_1_params}

    def init_canonical_params(self):
        '''Initialize canonical_params from cudnn_params blob.

           The cudnn_params blob is formatted as such: this may change
           depending on NVIDIA's wishes.
           ------------------------------------------------------------
           | weights                    | biases                      |
           ------------------------------------------------------------
           \                             \
            \                             \
             -------------------------------
             | layer1     |layer2     |... | (Only supporting 1 layer currently)
             -------------------------------
             \             \
              ---------------
              |fwd   |bak   |
              ---------------
              \       \
               \       \
                -----------------------------------------
                | Wi | Wf | Wc | Wo | Ri | Rf | Rc | Ro |
                -----------------------------------------
        '''
        self.canonical_params = {}

        for postfix in ['', '/Adam', '/Adam_1']:
            cudnn_params = self.cudnn_params[self.cudnn_params_name + postfix]

            # Get weights.
            shapes = (4 * [(self.n_units, self.n_input)] +
                      4 * [(self.n_units, self.n_units)])
            splits = np.cumsum([np.prod(shape) for shape in shapes])
            all_weights = cudnn_params[:self.num_dirs * self.cudnn_size_weight]
            for i, direction in enumerate(['fw_', 'bw_']):
                weights = all_weights[i*self.cudnn_size_weight:(i+1)*self.cudnn_size_weight]
                Wi, Wf, Wc, Wo, Ri, Rf, Rc, Ro, _ = np.split(weights, splits)
                params = [('Wi', Wi), ('Wf', Wf), ('Wc', Wc), ('Wo', Wo),
                          ('Ri', Ri), ('Rf', Rf), ('Rc', Rc), ('Ro', Ro)]
                params = {direction + name + postfix: value.reshape(shape)
                          for (name, value), shape in zip(params, shapes)}
                self.canonical_params.update(params)

            # Get biases.
            shapes = 8 * [(self.n_units,)]
            splits = np.cumsum([np.prod(shape) for shape in shapes])
            all_biases = cudnn_params[self.num_dirs * self.cudnn_size_weight:]
            for i, direction in enumerate(['fw_', 'bw_']):
                biases = all_biases[i*self.cudnn_size_bias:(i+1)*self.cudnn_size_bias]
                b_Wi, b_Wf, b_Wc, b_Wo, b_Ri, b_Rf, b_Rc, b_Ro, _ = np.split(biases, splits)
                params = [('b_Wi', b_Wi), ('b_Wf', b_Wf), ('b_Wc', b_Wc), ('b_Wo', b_Wo),
                          ('b_Ri', b_Ri), ('b_Rf', b_Rf), ('b_Rc', b_Rc), ('b_Ro', b_Ro)]
                params = {direction + name + postfix: value.reshape(shape)
                          for (name, value), shape in zip(params, shapes)}
                self.canonical_params.update(params)

    def init_basic_params(self):
        '''Initialize basic parameters. TODO: This is a lossy conversion!

        For each direction, TensorFlow has a separate parameter blob for the
        kernel and biases.

        A single kernel blob has shape (n_input + n_unit, 4 * n_unit).

        A single bias blob has shape (4 * n_unit)

        Both kernel and bias blobs use "icfo" gate order.
        '''
        self.basic_params = {}
        name = 'bidirectional_rnn/%s/basic_lstm_cell/%s%s'

        for postfix in ['', '/Adam', '/Adam_1']:
            for direction in ['fw', 'bw']:
                # Weights.
                kernel = np.array([]).reshape(0, self.n_input + self.n_units)
                for gate in ['i', 'c', 'f', 'o']:
                    input = self.canonical_params[direction + '_W' + gate + postfix]
                    hidden = self.canonical_params[direction + '_R' + gate + postfix]
                    combined = np.hstack([input, hidden])
                    kernel = np.vstack([kernel, combined])
                self.basic_params[name % (direction, 'kernel', postfix)] = kernel

                bias = np.array([])
                for gate in ['i', 'c', 'f', 'o']:
                    input = self.canonical_params[direction + '_b_W' + gate + postfix]
                    hidden = self.canonical_params[direction + '_b_R' + gate + postfix]
                    combined = input + hidden
                    bias = np.hstack([bias, combined])
                self.basic_params[name % (direction, 'bias', postfix)] = bias

    def add_non_lstm_values(self):
        '''Add extra values to params to simulate extra data in a checkpoint.'''
        def add_to_all(name, value):
            self.cudnn_params[name] = value
            self.canonical_params[name] = value
            self.basic_params[name] = value

        for i in range(5):
            shape = np.random.uniform(10, 20, (2,)).astype(int)
            add_to_all('extra_%d' % i, np.random.randn(*shape))

        add_to_all('global_step', np.array([10]))

    #- old_format -> Canonical tests ---------------------------------------------

    def test_canonical_to_canonical(self):
        '''Ensure canonical to canonical is the identity transform.'''
        # Defensive copy to ensure self canonical_params remains unchanged.
        new_params = copy.deepcopy(self.canonical_params)
        convert_params.to_canonical('canonical', new_params)
        self.parameter_equality(new_params, self.canonical_params)

    def test_cudnn_to_canonical(self):
        '''Check cudnn to canonical conversion.'''
        convert_params.to_canonical('cudnn', self.cudnn_params)
        self.parameter_equality(self.cudnn_params, self.canonical_params)

    def test_basic_to_canonical(self):
        '''TODO: This is a lossy conversion as canonical params two biases,
        basic only one.
        '''
        convert_params.to_canonical('basic', self.basic_params)

        # Fix expected canonical params to have zero hidden bias.
        for prefix in ['', '/Adam', '/Adam_1']:
            for direction in ['fw', 'bw']:
                for gate in ['i', 'c', 'f', 'o']:
                    input_name = direction + '_b_W' + gate + prefix
                    hidden_name = direction + '_b_R' + gate + prefix
                    hidden_value = self.canonical_params[hidden_name]
                    self.canonical_params[input_name] += hidden_value
                    self.canonical_params[hidden_name] = np.zeros(hidden_value.shape)

        self.parameter_equality(self.basic_params, self.canonical_params)

    #- Canonical -> new_format tests ---------------------------------------------

    def test_canonical_from_canonical(self):
        '''Ensure canonical to canonical is the identity transform.'''
        # Defensive copy to ensure self canonical_params remains unchanged.
        new_params = copy.deepcopy(self.canonical_params)
        convert_params.from_canonical('canonical', new_params)
        self.parameter_equality(new_params, self.canonical_params)

    def test_cudnn_from_canonical(self):
        '''Check canonical to cudnn conversion.'''
        convert_params.from_canonical('cudnn', self.canonical_params)
        self.parameter_equality(self.canonical_params, self.cudnn_params)

    def test_basic_from_canonical(self):
        '''Check canonical to basic conversion.'''
        convert_params.from_canonical('basic', self.canonical_params)
        self.parameter_equality(self.canonical_params, self.basic_params)

    #- Other -------------------------------------------------------------------

    def test_update_canonical_forget_bias(self):
        '''Check forget bias is added to forget gates only.'''
        for _ in range(10):
            bias_add = np.random.uniform(-10.0, 10.0)

            exp_canonical = copy.deepcopy(self.canonical_params)
            exp_canonical['fw_b_Wf'] += bias_add
            exp_canonical['bw_b_Wf'] += bias_add

            new_params = copy.deepcopy(self.canonical_params)
            convert_params.update_canonical_forget_bias(bias_add, new_params)

            self.parameter_equality(new_params, exp_canonical)

    def test_reset_global_step(self):
        values = {'global_step': 100}
        convert_params.reset_global_step(values)
        self.assertEqual(values['global_step'], 0)

    #- Helpers -----------------------------------------------------------------

    def parameter_equality(self, new_params, exp_params):
        '''Ensure new_params and exp_params are exactly equal.'''
        # Ensure nothing lost or added.
        self.assertListEqual(sorted(exp_params.keys()),
                             sorted(new_params.keys()))

        # Ensure all values unchanged.
        for name, exp_value in exp_params.items():
            if name[-4:] == 'Adam' or name[-6:] == 'Adam_1':
                continue
            self.assertIn(name, new_params)
            act_value = new_params[name]

            # Check dtype.
            self.assertEqual(act_value.dtype, exp_value.dtype,
                             msg=('%s dtype is not equal' % name))

            # Check shape.
            act_shape, exp_shape = act_value.shape, exp_value.shape
            self.assertEqual(act_shape, exp_shape,
                             msg=('%s shape is not equal. got: %s, exp: %s' % (name, act_shape, exp_shape)))

            # Check values.
            self.assertTrue(np.all(act_value == exp_value),
                            msg=('%s values do not match (%d, %d)' % (name, np.sum(act_value != exp_value), act_value.size)))

#-------------------------------------------------------------------------------
#    def test_opaque_to_canonical(self):
#        var_names_to_values = convert_params.opaque_to_canonical(self.params)
#
#        # Create TensorFlow graph and converter.
#        opaque_var = tf.get_variable('opaque_var', initializer=self.params)
#        converter = tf.contrib.cudnn_rnn.CudnnLSTMSaveable(opaque_var,
#                                                           num_layers=1,
#                                                           num_units=self.n_units,
#                                                           input_size=self.n_input,
#                                                           direction='bidirectional')
#
#        with tf.Session() as sess:
#            sess.run(tf.global_variables_initializer())
#
#            weights, biases = converter._OpaqueParamsToCanonical()
#
#            tf_vals = {}
#            # Forward.
#            tf_vals.update(cudnn_to_weights(weights[:8], prefix='fw_'))
#            tf_vals.update(cudnn_to_biases(biases[:8], prefix='fw_'))
#            # Backward.
#            tf_vals.update(cudnn_to_weights(weights[8:], prefix='bw_'))
#            tf_vals.update(cudnn_to_biases(biases[8:], prefix='bw_'))
#
#            for name, value in var_names_to_values.items():
#                tf_val = sess.run(tf_vals[name])
#                self.assertEqual(value.shape, tf_val.shape,
#                                 msg='np and tf vals have different shapes for %s' % name)
#                self.assertTrue(np.allclose(value, tf_val),
#                                msg='np and tf vals differ for %s' % name)
#
#def cudnn_to_weights(cu_weights, prefix=None):
#    '''Version of _cudnn_to_tf_weights to return a dict of name->Tensor.
#
#    https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/contrib/cudnn_rnn/python/ops/cudnn_rnn_ops.py#L495
#    '''
#
#    names = ['Wi', 'Wf', 'Wc', 'Wo',
#             'Ri', 'Rf', 'Rc', 'Ro']
#    if prefix is not None:
#        names = [prefix + name for name in names]
#    return dict(zip(names, cu_weights))
#
#def cudnn_to_biases(cu_biases, prefix=None):
#    '''Wrapper for _cudnn_to_tf_biases to return a dict of name->Tensor.
#
#    https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/contrib/cudnn_rnn/python/ops/cudnn_rnn_ops.py#L533
#    '''
#    names = ['b_Wi', 'b_Wf', 'b_Wc', 'b_Wo',
#             'b_Ri', 'b_Rf', 'b_Rc', 'b_Ro']
#    if prefix is not None:
#        names = [prefix + name for name in names]
#    return dict(zip(names, cu_biases))
