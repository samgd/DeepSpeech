import numpy as np
import re
import tensorflow as tf

from tensorflow.contrib.framework import assign_from_values

tf.app.flags.DEFINE_string ('in_ckpt',         '',      'checkpoint to read Tensor values from')
tf.app.flags.DEFINE_string ('out_ckpt',        '',      'checkpoint to write Tensor values to')
tf.app.flags.DEFINE_string ('in_type',         'basic', '')
tf.app.flags.DEFINE_string ('out_type',        'cudnn', '')
tf.app.flags.DEFINE_float  ('forget_bias_add', 0.0,     'value to add to forget gate Tensor - Adam Tensor values are not changed')

FLAGS = tf.app.flags.FLAGS

def main(_):
    var_names_to_values = get_tensors(FLAGS.in_ckpt)

    var_names_to_values = type_to_canonical(FLAGS.in_type, var_names_to_values)

    update_canonical_forget_bias(FLAGS.forget_bias_add, var_names_to_values)

    var_names_to_values = canonical_to_type(FLAGS.out_type, var_names_to_values)

    save_to_ckpt(FLAGS.out_ckpt, var_names_to_values)

#- Main functions --------------------------------------------------------------

def get_tensors(ckpt):
    '''Return a dict of name->value for all Tensors in the checkpoint.

    Args:
        ckpt: Filename of checkpoint to read Tensor values from.

    Returns:
        dict of name->value for Tensor in ckpt.
    '''
    reader = tf.train.NewCheckpointReader(ckpt)
    var_to_dtype_map = reader.get_variable_to_dtype_map()

    var_names_to_values = {}
    for name in var_to_dtype_map.items():
        var_names_to_values[name] = reader.get_tensor(name)
    return var_names_to_values

def type_to_canonical(old_type, var_names_to_values):
    '''Return an updated dict with the LSTM parameters in the canonical format.

    Args:
        old_type: Type of parameter set found in var_names_to_values. Can be
            any of ['basic', 'cudnn', 'canonical'].
        var_names_to_values: Dict of name->value.

    Returns:
        Updated name->value dict with the LSTM parameters converted to the
        canonical format.
    '''
    if old_type == 'canonical':
        return var_names_to_values

    if old_type == 'basic':
        split_p = re.compile('bidirectional_rnn/(fw|bw)/basic_lstm_cell/(kernel|bias)(/Adam|/Adam_1)?$')

        new_names_to_values = {}
        for name, value in var_names_to_values.items():
            match = split_p.search(name)

            if match is None:
                tensors = {name: value}
            elif match.group(2) == 'kernel':
                tensors = basic_to_canonical_weights(value,
                                                     prefix=match.group(1) +  '_',
                                                     postfix=match.group(3) or '')
            elif match.group(2) == 'bias':
                # Add forget bias to parameters only, not Adam.
                forget_bias = 0.0 # TODO: int(match.group(3) is None)
                tensors = basic_to_canonical_biases(value,
                                                    prefix=match.group(1) + '_',
                                                    postfix=match.group(3) or '',
                                                    forget_bias=forget_bias)
            new_names_to_values.update(tensors)
        var_names_to_values = new_names_to_values
    elif old_type == 'cudnn':
        cudnn_p = re.compile('fp32_storage/cudnn_lstm/opaque_kernel(/Adam(_1)?)?$')

        new_names_to_values = {}
        # Remove values that will be converted.
        for name, value in var_names_to_values.items():
            if cudnn_p.search(name):
                continue
            new_names_to_values[name] = value

        new_names_to_values.update(opaque_to_canonical(var_names_to_values))
        new_names_to_values.update(opaque_to_canonical(var_names_to_values, postfix='/Adam'))
        new_names_to_values.update(opaque_to_canonical(var_names_to_values, postfix='/Adam_1'))
        var_names_to_values = new_names_to_values
    else:
        raise ValueError('unknown old_type %r' % old_type)

    return var_names_to_values

def update_canonical_forget_bias(forget_bias_add, var_names_to_values):
    '''Add forget_bias_add to the forward and backward forget gate biases.

    Note: Adam values are not changed.

    Args:
        forget_bias_add: Value added to forget gate biases.
        var_names_to_values: Canonical parameter dictonary.
    '''
    var_names_to_values['fw_b_Wf'] += forget_bias_add
    var_names_to_values['bw_b_Wf'] += forget_bias_add

def canonical_to_type(new_type, var_names_to_values):
    '''Return an updated dict with the LSTM parameters in the given format.

    Args:
        new_type: Type of parameter set to convert the LSTM parameters in
            var_names_to_values to. Can be any of ['basic', 'cudnn', 'canonical'].
        var_names_to_values: Dict of name->value.

    Returns:
        Updated name->value dict with the LSTM parameters converted to the
        new_type format.
    '''
    if new_type == 'canonical':
        return var_names_to_values

    if new_type == 'cudnn':
        lstm_p = re.compile('(bw|fw)_(b_)?(R|W)(i|c|o|f)(/Adam|/Adam_1)?$')

        new_names_to_values = {}
        # Add values that will not be converted.
        for name, value in var_names_to_values.items():
            if lstm_p.search(name):
                continue
            new_names_to_values[name] = value

        kernel_name = 'fp32_storage/cudnn_lstm/opaque_kernel'
        new_names_to_values[kernel_name] = canonical_to_opaque(var_names_to_values)

        opaque = canonical_to_opaque(var_names_to_values, postfix='/Adam')
        new_names_to_values[kernel_name + '/Adam'] = opaque

        opaque = canonical_to_opaque(var_names_to_values, postfix='/Adam_1')
        new_names_to_values[kernel_name + '/Adam_1'] = opaque

        var_names_to_values = new_names_to_values
    elif new_type == 'basic':
        raise NotImplementedError
    else:
        raise ValueError('unknown new_type %r' % new_type)

    return var_names_to_values

def save_to_ckpt(ckpt, var_names_to_values):
    '''Save to out_ckpt.'''
    tf.reset_default_graph()
    for name, value in var_names_to_values.items():
        tf.get_variable(name, shape=value.shape, dtype=value.dtype)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        assign_op, feed_dict = assign_from_values(var_names_to_values)
        sess.run(assign_op, feed_dict)
        saver.save(sess, ckpt)

#- LSTM conversion: canonical->opaque ------------------------------------------

def canonical_to_opaque(var_names_to_values, postfix=''):
    '''
    '''
    fwd_weights = canonical_to_opaque_weights(var_names_to_values,
                                              prefix='fw_',
                                              postfix=postfix)
    bac_weights = canonical_to_opaque_weights(var_names_to_values,
                                              prefix='bw_',
                                              postfix=postfix)

    fwd_biases = canonical_to_opaque_biases(var_names_to_values,
                                            prefix='fw_',
                                            postfix=postfix)
    bac_biases = canonical_to_opaque_biases(var_names_to_values,
                                            prefix='bw_',
                                            postfix=postfix)

    opaque_params = np.concatenate([fwd_weights, bac_weights,
                                    fwd_biases, bac_biases])

    return opaque_params

def canonical_to_opaque_weights(var_names_to_values, n_units=2048, n_input=4096, prefix='', postfix=''):
    '''
    '''
    weight_shapes = get_weight_shapes(prefix, n_units, n_input, postfix)
    weights = np.array([])
    for name, _ in weight_shapes:
        values = var_names_to_values[name]
        weights = np.concatenate([weights, values.flatten()])
    return weights

def canonical_to_opaque_biases(var_names_to_values, prefix='', postfix=''):
    '''
    '''
    bias_names = get_bias_names(prefix, postfix)
    biases = np.array([])
    for name in bias_names:
        values = var_names_to_values[name]
        biases = np.concatenate([biases, values.flatten()])
    return biases

#- LSTM conversion: opaque->canonical ------------------------------------------

def opaque_to_canonical(var_names_to_values, n_units=2048, n_input=4096, postfix=''):
    '''TODO: Update this docstring

    Return a map of name->ndarray for each parameter in opaque_params.

    Args:
        opaque_params: Single-layer, bidirectional CudnnLSTM parameter blob.
        n_units: Number of units in CudnnLSTM.
        n_input: Input size to CudnnLSTM.

    Returns:
        Map of name->ndarray for each of the tensors in the opaque_params blob.
        The parameter names in the map are:

            ['Wi', 'Ri', 'b_Wi', 'b_Ri'
             'Wf', 'Rf', 'b_Wf', 'b_Rf'
             'Wc', 'Rc', 'b_Wc', 'b_Rc'
             'Wo', 'Ro', 'b_Wo', 'b_Ro']

        which correspond to the LSTM equations given by NVIDIA:

            it = sigm(Wi*x_t + Ri*h_t-1 + b_Wi + b_Ri)
            ft = sigm(Wf*x_t + Rf*h_t-1 + b_Wf + b_Rf)
            ot = sigm(Wo*x_t + Ro*h_t-1 + b_Wo + b_Ro)
            c't = tanh(Wc*x_t + Rc*h_t-1 + b_Wc + b_Rc)

        More info here:
            http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnRNNMode_t
    '''
    kernel_name = 'fp32_storage/cudnn_lstm/opaque_kernel' + postfix
    opaque_params = var_names_to_values[kernel_name]

    # Weights.
    fwd_bac_weight_split = 4*n_units*n_input + 4*n_units*n_units
    weight_end = 2*fwd_bac_weight_split
    fwd_weights = opaque_params[:fwd_bac_weight_split]
    bac_weights = opaque_params[fwd_bac_weight_split:weight_end]

    param_vals = {}
    param_vals.update(opaque_to_canonical_weights(fwd_weights, prefix='fw_', postfix=postfix))
    param_vals.update(opaque_to_canonical_weights(bac_weights, prefix='bw_', postfix=postfix))

    # Biases.
    fwd_bac_bias_split = 8*n_units
    fwd_biases = opaque_params[weight_end:weight_end+fwd_bac_bias_split]
    bac_biases = opaque_params[weight_end+fwd_bac_bias_split:]

    param_vals.update(opaque_to_canonical_biases(fwd_biases, prefix='fw_', postfix=postfix))
    param_vals.update(opaque_to_canonical_biases(bac_biases, prefix='bw_', postfix=postfix))

    return param_vals

def opaque_to_canonical_weights(weights, n_units=2048, n_input=4096, prefix='', postfix=''):
    '''Return a map of name->ndarray for each weight in weights.

    Args:
        weights: Single-layer, single-direction weight blob.
        prefix: Optional prefix to append to weight names.

    Returns:
        Map of name->ndarray for each weight.
    '''
    weight_shapes = get_weight_shapes(prefix, n_units, n_input, postfix)

    weight_vals = {}
    start = 0
    for name, shape in weight_shapes:
        end = start + np.prod(shape)
        weight = weights[start:end]
        weight_vals[name] = weight.reshape(shape)
        start = end

    return weight_vals

def opaque_to_canonical_biases(biases, n_units=2048, prefix='', postfix=''):
    '''Return a map of name->ndarray for each bias in biases.

    Args:
        biases: Single-layer, single-direction bias blob.
        prefix: Optional prefix to append to bias names.

    Returns:
        Map of name->ndarray for each bias.
    '''
    bias_names = get_bias_names(prefix, postfix)

    bias_vals = {}
    for i, name in enumerate(bias_names):
        bias = biases[i*n_units:(i + 1)*n_units]
        bias_vals[name] = bias

    return bias_vals

#- LSTM conversion: basic->canonical -------------------------------------------

def basic_to_canonical_weights(weights, n_units=2048, n_input=4096, prefix='', postfix=''):
    '''
    '''
    W_i, W_c, W_f, W_o = np.split(weights, 4, axis=1)

    w_i, r_i, _ = np.split(W_i, [n_input, n_input + n_units], axis=0)
    w_c, r_c, _ = np.split(W_c, [n_input, n_input + n_units], axis=0)
    w_f, r_f, _ = np.split(W_f, [n_input, n_input + n_units], axis=0)
    w_o, r_o, _ = np.split(W_o, [n_input, n_input + n_units], axis=0)

    params = {'Wi': w_i, 'Wf': w_f, 'Wc': w_c, 'Wo': w_o,
              'Ri': r_i, 'Rf': r_f, 'Rc': r_c, 'Ro': r_o}
    params = {prefix + name + postfix: value for name, value in params.items()}

    return params

def basic_to_canonical_biases(biases, n_units=2048, n_input=4096,
                              prefix='', postfix='', forget_bias=1.0):
    '''
    '''
    B_i, B_c, B_f, B_o = np.split(biases, 4, axis=0)

    shape = B_i.shape

    b_wi, b_ri = B_i, np.zeros(shape)
    b_wc, b_rc = B_c, np.zeros(shape)
    b_wf, b_rf = B_f + np.full(shape, forget_bias), np.zeros(shape)
    b_wo, b_ro = B_o, np.zeros(shape)

    params = {'b_Wi': b_wi, 'b_Wf': b_wf, 'b_Wc': b_wc, 'b_Wo': b_wo,
              'b_Ri': b_ri, 'b_Rf': b_rf, 'b_Rc': b_rc, 'b_Ro': b_ro}
    params = {prefix + name + postfix: value for name, value in params.items()}

    return params

#- LSTM conversion: canonical->basic -------------------------------------------

# TODO

#-------------------------------------------------------------------------------

def get_weight_shapes(prefix, n_units=2048, n_input=4096, postfix=''):
    '''
    '''
    weight_shapes = [('Wi', (n_input, n_units)),
                     ('Wf', (n_input, n_units)),
                     ('Wc', (n_input, n_units)),
                     ('Wo', (n_input, n_units)),
                     ('Ri', (n_units, n_units)),
                     ('Rf', (n_units, n_units)),
                     ('Rc', (n_units, n_units)),
                     ('Ro', (n_units, n_units))]
    weight_shapes = [(prefix + name + postfix, shape) for name, shape in weight_shapes]
    return weight_shapes

def get_bias_names(prefix, postfix=''):
    '''
    '''
    bias_names = ['b_Wi', 'b_Wf', 'b_Wc', 'b_Wo',
                  'b_Ri', 'b_Rf', 'b_Rc', 'b_Ro']
    bias_names = [prefix + name + postfix for name in bias_names]
    return bias_names

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    tf.app.run()
