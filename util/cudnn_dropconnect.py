import tensorflow as tf

from tensorflow.contrib.cudnn_rnn.python.layers import cudnn_rnn

_HIDDEN = 'hidden'
_INPUT = 'input'

def init_drop_types(hidden_keep_prob, input_keep_prob):
    dropconnect_types = []
    if hidden_keep_prob < 1.0:
       dropconnect_types += _HIDDEN
    if input_keep_prob < 1.0:
       dropconnect_types += _INPUT
    return dropconnect_types


def lstm(cudnn_lstm, hidden_keep_prob, input_keep_prob=1.0, drop_bias=False, seed=None):
    if not cudnn_lstm.built:
        raise ValueError('CudnnLSTM must be build before dropconnect is applied.')

    dropconnect_types = init_drop_types(hidden_keep_prob, input_keep_prob)
    dtype = cudnn_lstm.kernel.dtype

    weight_masks = []
    for shape, elem_type in _canonical_weight_shapes(cudnn_lstm):
        if elem_type not in dropconnect_types:
            weight_masks.append(tf.constant(1.0, dtype, shape))
            continue

        keep_prob = hidden_keep_prob if elem_type == _HIDDEN else input_keep_prob
        mask = _dropconnect_mask(shape, dtype, keep_prob, seed)
        weight_masks.append(mask)

    passthru_biases = [tf.ones(shape, dtype)
                       for shape in cudnn_lstm.canonical_bias_shapes]

    mask = cudnn_lstm._canonical_to_opaque(weight_masks, passthru_biases)
    cudnn_lstm.kernel = tf.multiply(mask, cudnn_lstm.kernel)


def _dropconnect_mask(shape, dtype, keep_prob, seed):
    # uniform [keep_prob, 1.0 + keep_prob]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(shape,
                                       seed=seed,
                                       dtype=dtype)
    # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
    return tf.floor(random_tensor)


# Wrap the shape properites to tag each shape with input or hidden type.
def _canonical_weight_shapes(cudnn):
    """Shapes of Cudnn canonical weight tensors."""
    if not cudnn._input_size:
        raise RuntimeError(
            "%s.canonical_weight_shapes invoked before input shape is known" %
            type(cudnn).__name__)

    shapes = []
    for i in range(cudnn._num_layers):
        shapes.extend(_canonical_weight_shape(cudnn, i))
    return shapes

def _canonical_weight_shape(cudnn, layer):
    """Shapes of Cudnn canonical weight tensors for given layer."""
    if layer < 0 or layer >= cudnn._num_layers:
        raise ValueError("\'layer\' is not valid, got %s, expecting [%d, %d]" %
                         (layer, 0, cudnn._num_layers-1))
    if not cudnn._input_size:
        raise RuntimeError(
            "%s._canonical_weight_shape invoked before input shape is known" %
            type(cudnn).__name__)

    input_size = cudnn._input_size
    num_units = cudnn._num_units
    num_gates = cudnn._num_params_per_layer // 2
    is_bidi = cudnn._direction == cudnn_rnn.CUDNN_RNN_BIDIRECTION

    if layer == 0:
        wts_applied_on_inputs = [(num_units, input_size)] * num_gates
    else:
        if is_bidi:
            wts_applied_on_inputs = [(num_units, 2 * num_units)] * num_gates
        else:
            wts_applied_on_inputs = [(num_units, num_units)] * num_gates

    wts_applied_on_hidden_states = [(num_units, num_units)] * num_gates

    wts_applied_on_inputs = [(shape, _INPUT) for shape in wts_applied_on_inputs]
    wts_applied_on_hidden_states = [(shape, _HIDDEN) for shape in wts_applied_on_hidden_states]

    tf_wts = wts_applied_on_inputs + wts_applied_on_hidden_states
    return tf_wts if not is_bidi else tf_wts * 2
