import numpy as np
import tensorflow as tf

def join(var_names_to_values):
    fwd_weights = join_opaque_weights(var_names_to_values, prefix='fw_')
    bac_weights = join_opaque_weights(var_names_to_values, prefix='bw_')

    fwd_biases = join_opaque_biases(var_names_to_values, prefix='fw_')
    bac_biases = join_opaque_biases(var_names_to_values, prefix='bw_')

    opaque_params = np.concatenate([fwd_weights, bac_weights,
                                    fwd_biases, bac_biases])

    return opaque_params

def join_opaque_weights(var_names_to_values, n_units=2048, n_input=4096, prefix=''):
    weight_shapes = get_weight_shapes(prefix, n_units, n_input)
    weights = np.array([])
    for name, shape in weight_shapes:
        values = var_names_to_values[name]
        weights = np.concatenate([weights, values.flatten()])
    return weights

def join_opaque_biases(var_names_to_values, prefix=''):
    bias_names = get_bias_names(prefix)
    biases = np.array([])
    for name in bias_names:
        values = var_names_to_values[name]
        biases = np.concatenate([biases, values.flatten()])
    return biases

def split(opaque_params, n_units=2048, n_input=4096):
    '''Return a map of name->ndarray for each parameter in opaque_params.

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
    # Weights.
    fwd_bac_weight_split = 4*n_units*n_input + 4*n_units*n_units
    weight_end = 2*fwd_bac_weight_split
    fwd_weights = opaque_params[:fwd_bac_weight_split]
    bac_weights = opaque_params[fwd_bac_weight_split:weight_end]

    param_vals = {}
    param_vals.update(split_opaque_weights(fwd_weights, prefix='fw_'))
    param_vals.update(split_opaque_weights(bac_weights, prefix='bw_'))

    # Biases.
    fwd_bac_bias_split = 8*n_units
    fwd_biases = opaque_params[weight_end:weight_end+fwd_bac_bias_split]
    bac_biases = opaque_params[weight_end+fwd_bac_bias_split:]

    param_vals.update(split_opaque_biases(fwd_biases, prefix='fw_'))
    param_vals.update(split_opaque_biases(bac_biases, prefix='bw_'))

    return param_vals

def split_opaque_weights(weights, n_units=2048, n_input=4096, prefix=''):
    '''Return a map of name->ndarray for each weight in weights.

    Args:
        weights: Single-layer, single-direction weight blob.
        prefix: Optional prefix to append to weight names.

    Returns:
        Map of name->ndarray for each weight.
    '''
    weight_shapes = get_weight_shapes(prefix, n_units, n_input)

    weight_vals = {}
    start = 0
    for name, shape in weight_shapes:
        end = start + np.prod(shape)
        weight = weights[start:end]
        weight_vals[name] = weight.reshape(shape)
        start = end

    return weight_vals

def split_opaque_biases(biases, n_units=2048, prefix=''):
    '''Return a map of name->ndarray for each bias in biases.

    Args:
        biases: Single-layer, single-direction bias blob.
        prefix: Optional prefix to append to bias names.

    Returns:
        Map of name->ndarray for each bias.
    '''
    bias_names = get_bias_names(prefix)

    bias_vals = {}
    for i, name in enumerate(bias_names):
        bias = biases[i*n_units:(i + 1)*n_units]
        bias_vals[name] = bias

    return bias_vals

def get_weight_shapes(prefix, n_units=2048, n_input=4096):
    weight_shapes = [('Wi', (n_units, n_input)),
                     ('Wf', (n_units, n_input)),
                     ('Wc', (n_units, n_input)),
                     ('Wo', (n_units, n_input)),
                     ('Ri', (n_units, n_units)),
                     ('Rf', (n_units, n_units)),
                     ('Rc', (n_units, n_units)),
                     ('Ro', (n_units, n_units))]
    weight_shapes = [(prefix + name, shape) for name, shape in weight_shapes]
    return weight_shapes

def get_bias_names(prefix):
    bias_names = ['b_Wi', 'b_Wf', 'b_Wc', 'b_Wo',
                  'b_Ri', 'b_Rf', 'b_Rc', 'b_Ro']
    bias_names = [prefix + name for name in bias_names]
    return bias_names
