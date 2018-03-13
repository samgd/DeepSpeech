'''Convert a single-layer bidirectional LSTM in a TensorFlow checkpoint from
one format to another where the possible formats are:

    'basic': TensorFlow format (i.e. BasicLSTM)
    'cudnn': NVIDIA format (i.e. CudnnLSTM)
    'canonical': Canonical format.


Given an input size of 'n_input' and a hidden size of 'n_unit':

'basic' format LSTM parameter name and shapes:

    - 'bidirectional_rnn/fw/basic_lstm_cell/kernel'
    - 'bidirectional_rnn/bw/basic_lstm_cell/kernel'
    - 'bidirectional_rnn/fw/basic_lstm_cell/bias'
    - 'bidirectional_rnn/bw/basic_lstm_cell/bias'

    - Kernel shape: (n_input + n_unit, 4 * n_unit)
        - Input and hidden state vectors are stacked multiplied by this kernel
          to compute part of the new (input, c/gate, forget, output)
          pre-activation gate values.
    - Bias shape: (4 * n_unit,)
        - A set of biases for each gate, stacked into one vector, used to
          compute the new pre-activation gate values.

'cudnn' format LSTM parameter names and shapes:

    - 'cudnn_lstm/opaque_kernel'

    - Shape: (2 * (4 * (n_input * n_unit) + 4 * (n_unit * n_unit) + 8 * n_unit),)
        - Two directions (forward and backward) where the new pre-activation
          gate values (input, forget, c/gate, output) are computed using 4
          matrices acting on the input vector, 4 matrices acting on the hidden
          state, and 8 biases (two for each of the four gates).

'canonical' format LSTM parameter names and shapes:

    - 'fw_Wi', 'fw_Wf', 'fw_Wc', 'fw_Wo'
    - 'fw_Ri', 'fw_Rf', 'fw_Rc', 'fw_Ro'
    - 'bw_Wi', 'bw_Wf', 'bw_Wc', 'bw_Wo'
    - 'bw_Ri', 'bw_Rf', 'bw_Rc', 'bw_Ro'

    - 'fw_b_Wi', 'fw_b_Wf', 'fw_b_Wc', 'fw_b_Wo'
    - 'fw_b_Ri', 'fw_b_Rf', 'fw_b_Rc', 'fw_b_Ro'
    - 'bw_b_Wi', 'bw_b_Wf', 'bw_b_Wc', 'bw_b_Wo'
    - 'bw_b_Ri', 'bw_b_Rf', 'bw_b_Rc', 'bw_b_Ro'

    - Forward + Backward Input Kernel Shape ([fb]w_W[ifco]): (n_unit, n_input)
    - Forward + Backward Hidden Kernel Shape ([fb]w_R[ifco]): (n_unit, n_unit)
    - Forward + Backward Input Bias Shape ([fb]w_b_W[ifco]): (n_unit,)
    - Forward + Backward Input Bias Shape ([fb]w_b_R[ifco]): (n_unit,)


If Adam is used as the optimizer Tensors with names equal to the above but with
both '/Adam' and '/Adam_1' appended will also be present.


The 'basic' format has a single bias for each of the 4 gates (input, forget,
output, c/gate) whereas the 'cudnn' and 'canonical' formats have two for each
gate.  Converting to the 'basic' format is thus a lossy operation - the total
bias value is equal but the relative split information is lost.


The BasicLSTM in TensorFlow, which utilizes the 'basic' parameter format, has a
function parameter that allows a bias to be added to the forget gate during
run-time.  The CudnnLSTM RNN, using the 'cudnn' format, does not have this
and thus the desired forget gate bias must be added to the forget gate values
stored in the checkpoint before use. A flag is provided to do this.
'''

import numpy as np
import re
import tensorflow as tf

from tensorflow.contrib.framework import assign_from_values

tf.app.flags.DEFINE_string ('in_ckpt',         '',      'checkpoint to read Tensor values from')
tf.app.flags.DEFINE_string ('out_ckpt',        '',      'checkpoint to write Tensor values to')
tf.app.flags.DEFINE_string ('in_format',       'basic', '')
tf.app.flags.DEFINE_string ('out_format',      'cudnn', '')
tf.app.flags.DEFINE_float  ('forget_bias_add', 0.0,     'value to add to forget gate Tensor - Adam Tensor values are not changed')

FLAGS = tf.app.flags.FLAGS

def main(_):
    var_names_to_values = get_tensors(FLAGS.in_ckpt)
    to_canonical(FLAGS.in_format, var_names_to_values)
    update_canonical_forget_bias(FLAGS.forget_bias_add, var_names_to_values)
    from_canonical(FLAGS.out_format, var_names_to_values)
    save_to_ckpt(FLAGS.out_ckpt, var_names_to_values)

def get_tensors(ckpt):
    '''Return a dictionary of name: value for all Tensors in the checkpoint.

    Args:
        ckpt: Filename of TensorFlow checkpoint to read Tensor values from.

    Returns:
        Dictionary of Tensor name: value.
    '''
    reader = tf.train.NewCheckpointReader(ckpt)
    var_to_dtype_map = reader.get_variable_to_dtype_map()

    var_names_to_values = {}
    for name in var_to_dtype_map:
        var_names_to_values[name] = reader.get_tensor(name)
    return var_names_to_values

def to_canonical(old_format, var_names_to_values):
    '''Convert dictionary contents from one format to 'canonical'.

    Args:
        var_names_to_values: Name: value dictionary.
        old_format: Format of parameter set found in var_names_to_values. Can
            be any of ['basic', 'cudnn', 'canonical'].
    '''
    if old_format == 'canonical':
        return

    if old_format == 'basic':
        split_p = re.compile('bidirectional_rnn/([fb]w)/basic_lstm_cell/(kernel|bias)(/Adam|/Adam_1)?$')

        new_names_to_values = {}
        for name, value in var_names_to_values.items():
            match = split_p.search(name)

            if match is None:
                tensors = {name: value}
            elif match.group(2) == 'kernel':
                tensors = basic_to_canonical_weights(value,
                                                     prefix=match.group(1) + '_',
                                                     postfix=match.group(3) or '')
            elif match.group(2) == 'bias':
                tensors = basic_to_canonical_biases(value,
                                                    prefix=match.group(1) + '_',
                                                    postfix=match.group(3) or '')
            new_names_to_values.update(tensors)
        var_names_to_values.clear()
        var_names_to_values.update(new_names_to_values)
    elif old_format == 'cudnn':
        cudnn_p = re.compile('(?:fp32_storage/)cudnn_lstm/opaque_kernel(/Adam(?:_1)?)?$')

        new_names_to_values = {}
        # Remove values that will be converted.
        for name, value in var_names_to_values.items():
            if cudnn_p.search(name):
                continue
            new_names_to_values[name] = value

        new_names_to_values.update(cudnn_to_canonical(var_names_to_values))
        new_names_to_values.update(cudnn_to_canonical(var_names_to_values, postfix='/Adam'))
        new_names_to_values.update(cudnn_to_canonical(var_names_to_values, postfix='/Adam_1'))
        var_names_to_values.clear()
        var_names_to_values.update(new_names_to_values)
    else:
        raise ValueError('unknown old_format %r' % old_format)

def update_canonical_forget_bias(forget_bias_add, var_names_to_values):
    '''Add forget_bias_add to the forward and backward forget gate biases.

    Adam values are not changed.

    Args:
        forget_bias_add: Value added to forget gate biases.
        var_names_to_values: Name: value dictionary.
    '''
    var_names_to_values['fw_b_Wf'] += forget_bias_add
    var_names_to_values['bw_b_Wf'] += forget_bias_add

def from_canonical(new_format, var_names_to_values):
    '''Convert dictionary contents from 'canonical' to a new format.

    Args:
        new_format: Formatt of parameter set to convert the LSTM parameters in
            var_names_to_values to. Can be any of ['basic', 'cudnn', 'canonical'].
        var_names_to_values: Name: value dictonary.
    '''
    if new_format == 'canonical':
        return

    if new_format == 'cudnn':
        lstm_p = re.compile('(bw|fw)_(b_)?(R|W)([icof])(/Adam|/Adam_1)?$')

        new_names_to_values = {}
        # Add values that will not be converted.
        for name, value in var_names_to_values.items():
            if lstm_p.search(name):
                continue
            new_names_to_values[name] = value

        kernel_name = 'cudnn_lstm/opaque_kernel'
        new_names_to_values[kernel_name] = canonical_to_cudnn(var_names_to_values)

        cudnn = canonical_to_cudnn(var_names_to_values, postfix='/Adam')
        new_names_to_values[kernel_name + '/Adam'] = cudnn

        cudnn = canonical_to_cudnn(var_names_to_values, postfix='/Adam_1')
        new_names_to_values[kernel_name + '/Adam_1'] = cudnn

        var_names_to_values.clear()
        var_names_to_values.update(new_names_to_values)
    elif new_format == 'basic':
        raise NotImplementedError
    else:
        raise ValueError('unknown new_format %r' % new_format)

def save_to_ckpt(ckpt, var_names_to_values):
    '''Save to dictionary contents to a TensorFlow checkpoint.

    Args:
        ckpt: Checkpoint path.
        var_names_to_values: Name: value dictionary.
    '''
    tf.reset_default_graph()
    for name, value in var_names_to_values.items():
        tf.get_variable(name, shape=value.shape, dtype=value.dtype)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        assign_op, feed_dict = assign_from_values(var_names_to_values)
        sess.run(assign_op, feed_dict)
        saver.save(sess, ckpt)

#- canonical -> cudnn ----------------------------------------------------------

def canonical_to_cudnn(var_names_to_values, postfix=''):
    '''Convert canonical params to cudnn format and return in a new dictionary.

    Args:
        var_names_to_values: Name: value dictionary.
        postfix: String to append to each TensorName in the new dictionary.

    Returns:
        Name: value dictionary.
    '''
    fwd_weights = canonical_to_cudnn_weights(var_names_to_values,
                                             prefix='fw_',
                                             postfix=postfix)
    bac_weights = canonical_to_cudnn_weights(var_names_to_values,
                                             prefix='bw_',
                                             postfix=postfix)

    fwd_biases = canonical_to_cudnn_biases(var_names_to_values,
                                           prefix='fw_',
                                           postfix=postfix)
    bac_biases = canonical_to_cudnn_biases(var_names_to_values,
                                           prefix='bw_',
                                           postfix=postfix)

    cudnn_params = np.concatenate([fwd_weights, bac_weights,
                                   fwd_biases, bac_biases])

    return cudnn_params

def canonical_to_cudnn_weights(var_names_to_values, n_units=2048, n_input=4096,
                               prefix='', postfix=''):
    '''Convert canonical weights to cudnn format.'''
    weight_shapes = get_weight_shapes(prefix, n_units, n_input, postfix)
    weights = np.array([])
    for name, _ in weight_shapes:
        values = var_names_to_values[name]
        weights = np.concatenate([weights, values.flatten()])
    return weights

def canonical_to_cudnn_biases(var_names_to_values, prefix='', postfix=''):
    '''Convert canonical biases to cudnn format.'''
    bias_names = get_bias_names(prefix, postfix)
    biases = np.array([])
    for name in bias_names:
        values = var_names_to_values[name]
        biases = np.concatenate([biases, values.flatten()])
    return biases

#- cudnn -> canonical ----------------------------------------------------------

def cudnn_to_canonical(var_names_to_values, n_units=2048, n_input=4096, postfix=''):
    '''Convert cudnn params to canonical format and return in a new dictionary.

    Args:
        var_names_to_values: Name: value dictionary.
        postfix: String to append to each TensorName in the new dictionary.

    Returns:
        Name: value dictionary.
    '''
    kernel_name = 'fp32_storage/cudnn_lstm/opaque_kernel' + postfix
    cudnn_params = var_names_to_values[kernel_name]

    # Weights.
    fwd_bac_weight_split = 4*n_units*n_input + 4*n_units*n_units
    weight_end = 2*fwd_bac_weight_split
    fwd_weights = cudnn_params[:fwd_bac_weight_split]
    bac_weights = cudnn_params[fwd_bac_weight_split:weight_end]

    param_vals = {}
    param_vals.update(cudnn_to_canonical_weights(fwd_weights, prefix='fw_', postfix=postfix))
    param_vals.update(cudnn_to_canonical_weights(bac_weights, prefix='bw_', postfix=postfix))

    # Biases.
    fwd_bac_bias_split = 8*n_units
    fwd_biases = cudnn_params[weight_end:weight_end+fwd_bac_bias_split]
    bac_biases = cudnn_params[weight_end+fwd_bac_bias_split:]

    param_vals.update(cudnn_to_canonical_biases(fwd_biases, prefix='fw_', postfix=postfix))
    param_vals.update(cudnn_to_canonical_biases(bac_biases, prefix='bw_', postfix=postfix))

    return param_vals

def cudnn_to_canonical_weights(weights, n_units=2048, n_input=4096, prefix='', postfix=''):
    '''Convert cudnn weights to canonical format.'''
    weight_shapes = get_weight_shapes(prefix, n_units, n_input, postfix)

    weight_vals = {}
    start = 0
    for name, shape in weight_shapes:
        end = start + np.prod(shape)
        weight = weights[start:end]
        weight_vals[name] = weight.reshape(shape)
        start = end

    return weight_vals

def cudnn_to_canonical_biases(biases, n_units=2048, prefix='', postfix=''):
    '''Convert cudnn biases to canonical format.'''
    bias_names = get_bias_names(prefix, postfix)

    bias_vals = {}
    for i, name in enumerate(bias_names):
        bias = biases[i*n_units:(i + 1)*n_units]
        bias_vals[name] = bias

    return bias_vals

#- basic -> canonical ----------------------------------------------------------

def basic_to_canonical_weights(weights, n_units=2048, n_input=4096, prefix='', postfix=''):
    '''Convert basic weights to canonical format.'''
    if weights.shape[0] != 4 * n_units:
        weights = weights.T
    assert weights.shape == (4 * n_units, n_input + n_units)

    W_i, W_c, W_f, W_o = np.split(weights, 4, axis=0)

    w_i, r_i, _ = np.split(W_i, [n_input, n_input + n_units], axis=1)
    w_c, r_c, _ = np.split(W_c, [n_input, n_input + n_units], axis=1)
    w_f, r_f, _ = np.split(W_f, [n_input, n_input + n_units], axis=1)
    w_o, r_o, _ = np.split(W_o, [n_input, n_input + n_units], axis=1)

    params = {'Wi': w_i, 'Wf': w_f, 'Wc': w_c, 'Wo': w_o,
              'Ri': r_i, 'Rf': r_f, 'Rc': r_c, 'Ro': r_o}
    params = {prefix + name + postfix: value for name, value in params.items()}

    return params

def basic_to_canonical_biases(biases, n_units=2048, n_input=4096, prefix='', postfix=''):
    '''Convert basic biases to canonical format.'''
    B_i, B_c, B_f, B_o = np.split(biases, 4, axis=0)

    shape = B_i.shape

    b_wi, b_ri = B_i, np.zeros(shape)
    b_wc, b_rc = B_c, np.zeros(shape)
    b_wf, b_rf = B_f, np.zeros(shape)
    b_wo, b_ro = B_o, np.zeros(shape)

    params = {'b_Wi': b_wi, 'b_Wf': b_wf, 'b_Wc': b_wc, 'b_Wo': b_wo,
              'b_Ri': b_ri, 'b_Rf': b_rf, 'b_Rc': b_rc, 'b_Ro': b_ro}
    params = {prefix + name + postfix: value for name, value in params.items()}

    return params

#- TODO: canonical -> basic ----------------------------------------------------

#-------------------------------------------------------------------------------

def get_weight_shapes(prefix, n_units=2048, n_input=4096, postfix=''):
    '''Return a list of (name, shape) for each canonical weight Tensor.'''
    weight_shapes = [('Wi', (n_units, n_input)),
                     ('Wf', (n_units, n_input)),
                     ('Wc', (n_units, n_input)),
                     ('Wo', (n_units, n_input)),
                     ('Ri', (n_units, n_units)),
                     ('Rf', (n_units, n_units)),
                     ('Rc', (n_units, n_units)),
                     ('Ro', (n_units, n_units))]
    weight_shapes = [(prefix + name + postfix, shape) for name, shape in weight_shapes]
    return weight_shapes

def get_bias_names(prefix, postfix=''):
    '''Return a list of names for each canonical bias Tensor.'''
    bias_names = ['b_Wi', 'b_Wf', 'b_Wc', 'b_Wo',
                  'b_Ri', 'b_Rf', 'b_Rc', 'b_Ro']
    bias_names = [prefix + name + postfix for name in bias_names]
    return bias_names

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    tf.app.run()
