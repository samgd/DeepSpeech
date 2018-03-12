import numpy as np
import tensorflow as tf

from tensorflow.contrib.framework import assign_from_values

from util.convert_params import to_canonical

def get_mask_name(tensor_name):
    '''Return the name of the mask for a given a Tensor.'''
    return tensor_name + '/mask'

def apply_masks(out_ckpt, in_ckpt):
    '''Applies masks to Tensors with masks in in_ckpt.

    Masks are not saved in out_ckpt.

    Args:
        out_ckpt: Checkpoint to write Tensor values to.
        in_ckpt: Checkpoint to read Tensor and mask values from.
    '''
    tf.reset_default_graph()

    reader = tf.train.NewCheckpointReader(in_ckpt)
    var_to_shape_map = reader.get_variable_to_shape_map()
    var_to_dtype_map = reader.get_variable_to_dtype_map()
    var_names_to_values = {}

    for name, shape in var_to_shape_map.items():
        # Create variable in out_ckpt and add to value map.
        tensor = reader.get_tensor(name)

        # Read mask and apply mask if exist.
        mask_name = get_mask_name(name)
        if mask_name in var_to_shape_map:
            mask = reader.get_tensor(mask_name)
            tensor = np.multiply(tensor, mask)

        dtype = var_to_dtype_map[name]
        tf.get_variable(name, shape=shape, dtype=dtype)
        var_names_to_values[name] = tensor

    saver = tf.train.Saver()
    with tf.Session() as sess:
        assign_op, feed_dict = assign_from_values(var_names_to_values)
        sess.run(assign_op, feed_dict)
        saver.save(sess, out_ckpt)

def tensor_sparsity_percent(tensor):
    '''Returns sparsity % of a given Tensor.'''
    return (float(np.sum(tensor == 0)) / tensor.size) * 100

def ckpt_layerwise_sparsity_percent(ckpt, to_mask, cudnn_params_name=''):
    '''Return the sparsity percentage of each name in to_mask.'''
    reader = tf.train.NewCheckpointReader(ckpt)
    var_to_shape_map = reader.get_variable_to_shape_map()

    sparsity_percents = {}
    for name in var_to_shape_map:
        if name == cudnn_params_name:
            mask = reader.get_tensor(get_mask_name(name))
            param_vals = to_canonical(mask, 'cudnn')
            for name, value in param_vals.items():
                sparsity_percents[name] = tensor_sparsity_percent(value)
        elif name in to_mask:
            mask_name = get_mask_name(name)
            mask = reader.get_tensor(mask_name)
            sparsity_percents[name] = tensor_sparsity_percent(mask)

    return sparsity_percents

def ckpt_sparsity_percent(ckpt, to_mask):
    '''Return total sparsity percentage of Tensors in to_mask.

    Args:
        out_ckpt: Checkpoint to read mask values from.
        to_mask: Names of Tensors to include in sparsity calculation.

    Returns:
        Sparsity value [0.0, 100.0].
    '''
    reader = tf.train.NewCheckpointReader(ckpt)
    var_to_shape_map = reader.get_variable_to_shape_map()

    total_params = 0
    total_masked = 0
    for name in to_mask:
        tensor = reader.get_tensor(name)
        total_params += tensor.size

        mask_name = get_mask_name(name)
        if mask_name not in var_to_shape_map:
            continue
        mask = reader.get_tensor(mask_name)
        total_masked += np.sum(mask == 0)

    return float(total_masked) / total_params * 100.0
