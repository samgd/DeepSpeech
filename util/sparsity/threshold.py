import numpy as np
import tensorflow as tf
import sys

import util.opaque_params as opaque_params

from tensorflow.contrib.framework import assign_from_values

from util.sparsity.mask import get_mask_name
from util.sparsity.mask import tensor_sparsity_percent

def layerwise_add_masks_limit(out_ckpt, in_ckpt, to_mask, limit_sparsity=0.0,
                              opaque_params_name=''):
    '''Add masks for variables in to_mask to out_ckpt up to limit_sparsity.'''
    # Get names of variables whose sparsity is less than limit_sparsity.
    reader = tf.train.NewCheckpointReader(in_ckpt)
    var_to_shape_map = reader.get_variable_to_shape_map()

    to_increase = []
    for name in to_mask:
        mask_name = get_mask_name(name)

        if mask_name not in var_to_shape_map:
            # No mask, must create one.
            to_increase.append(name)
            continue

        sparsity = tensor_sparsity_percent(reader.get_tensor(mask_name))
        if sparsity < limit_sparsity:
            to_increase.append(name)

    add_masks(out_ckpt,
              in_ckpt,
              to_increase,
              sparsity=limit_sparsity,
              use_layerwise_threshold=True,
              opaque_params_name=opaque_params_name)

def add_masks(out_ckpt, in_ckpt, to_mask, sparsity=0.0,
              use_layerwise_threshold=False,
              opaque_params_name=''):
    '''Add masks for variables in to_mask to out_ckpt based on sparsity percent.

    The sparsity of masks already present in in_ckpt is taken into account -
    the current sparsity percentage is computed and subtracted from the desired
    sparsity percentage. A new mask is then computed whilst ensuring the
    current mask is preserved if present.

    Args:
        out_ckpt: Checkpoint to write Tensor and mask values to.
        in_ckpt: Checkpoint to read Tensor values from.
        to_mask: Tensor names to add or increase sparsity masks to.
        sparsity: Sparsity percentage between 0.0 and 100.0 inclusive.
            (default 0.0)
        use_layerwise_threshold: If True compute threshold layerwise rather
            than globally. This ensures that each layer is sparsity percent
            sparse rather than just the network as a whole.
    '''
    var_names_to_values = get_values(in_ckpt, to_mask)
    if use_layerwise_threshold and opaque_params_name:
        var_names_to_values = split_opaque(var_names_to_values, opaque_params_name)
    if not use_layerwise_threshold:
        thresholds = global_threshold(var_names_to_values, to_mask, sparsity)
    else:
        thresholds = layerwise_threshold(var_names_to_values, to_mask, sparsity)
    var_names_to_values = update_masks(var_names_to_values, to_mask, thresholds)
    if use_layerwise_threshold and opaque_params_name:
        var_names_to_values = join_opaque(var_names_to_values, opaque_params_name)
    create_graph(var_names_to_values)
    save_to_checkpoint(out_ckpt, var_names_to_values)

def save_to_checkpoint(out_ckpt, var_names_to_values):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        assign_op, feed_dict = assign_from_values(var_names_to_values)
        sess.run(assign_op, feed_dict)
        saver.save(sess, out_ckpt)

def create_graph(var_names_to_values):
    tf.reset_default_graph()
    for name, value in var_names_to_values.items():
        tf.get_variable(name, shape=value.shape)

def update_masks(var_names_to_values, to_mask, thresholds):
    updated_values = {}
    all_mask_names = [get_mask_name(name) for name in to_mask]

    for name, tensor in var_names_to_values.items():
        if name in all_mask_names:
            # Is a mask, skip.
            continue
        updated_values[name] = tensor

        if name not in to_mask:
            continue

        mask_name = get_mask_name(name)
        current_mask = var_names_to_values[mask_name]
        tensor = np.multiply(tensor, current_mask)
        updated_values[mask_name] = np.abs(tensor) > thresholds[name]
    return updated_values

def get_values(ckpt, to_mask):
    '''Return a dict of Tensor names in to_mask, and mask names, to values.'''
    reader = tf.train.NewCheckpointReader(ckpt)
    var_to_shape_map = reader.get_variable_to_shape_map()

    var_names_to_values = {}
    for name in var_to_shape_map:
        # Get Tensor and get or create mask.
        tensor = reader.get_tensor(name)
        var_names_to_values[name] = tensor

        if name not in to_mask:
            continue

        mask_name = get_mask_name(name)
        if mask_name in var_to_shape_map:
            mask = reader.get_tensor(mask_name)
        else:
            mask = np.ones(tensor.shape)
        # Store values to manipulate later.
        var_names_to_values[mask_name] = mask

    return var_names_to_values

def split_opaque(var_names_to_values, opaque_params_name):
    updated_values = {}

    params = var_names_to_values[opaque_params_name]
    updated_values.update(opaque_params.split(params))

    mask = var_names_to_values[get_mask_name(opaque_params_name)]
    split_mask = opaque_params.split(mask)
    split_mask = {get_mask_name(name): val for name, val in split_mask.items()}
    updated_values.update(split_mask)

    for name, value in var_names_to_values.items():
        if name == opaque_params_name:
            continue
        updated_values[name] = value

    return updated_values

def join_opaque(var_names_to_values, opaque_params_name):
    updated_values = {}
    # Reconstruct parameter blob.
    weight_shapes = opaque_params.get_weight_shapes('fw_')
    weight_shapes.extend(opaque_params.get_weight_shapes('bw_'))
    weight_names = [name for name, _ in weight_shapes]
    split_params = {name: var_names_to_values[name]
                    for name in weight_names}
    updated_values[opaque_params_name] = opaque_params.join(split_params)
    # Reconstruct parameter blob mask.
    bias_names = opaque_params.get_bias_names('fw_')
    bias_names.extend(opaque_params.get_bias_names('bw_'))
    split_mask = {name: var_names_to_values[get_mask_name(name)]
                  for name in bias_names}
    mask = opaque_params.join(split_mask)
    updated_values[get_mask_name(opaque_params_name)] = mask
    # Add remaining variables to new dict.
    for name, value in var_names_to_values.items():
        if name in weight_names or name in bias_names:
            continue
        updated_values[name] = value
    return updated_values

def layerwise_threshold(var_names_to_values, to_mask, sparsity=0.0):
    '''Compute the layerwise threshold values for the given sparsity level.

    The threshold value for each Tensor in to_mask is the `sparsity'th
    percentile of its absolute values.

    The sparsity of masks already present in in_ckpt is taken into account -
    the current sparsity percentage is computed and subtracted from the desired
    sparsity percentage, the Tensors have their corresponding mask applied if
    present, and then a new threshold value is computed.

    Args:
        in_ckpt: Checkpoint to read Tensor values from.
        to_mask: Tensor names to compute threshold values for.
        sparsity: Sparsity percentage between 0.0 and 100.0 inclusive.
            (default 0.0)

    Returns:
        Dictionary of {name, threshold_value for name in to_mask}.
    '''
    thresholds = {}
    for name, tensor in var_names_to_values.items():
        if name not in to_mask:
            # Compute thresholds for Tensors only.
            continue
        mask = var_names_to_values[get_mask_name(name)]
        thresholds[name] = tensor_threshold(tensor, sparsity, mask=mask)
    return thresholds

def global_threshold(var_names_to_values, to_mask, sparsity=0.0):
    '''Compute the global threshold value for the given sparsity level.

    The threshold value is the `sparsity'th percentile of the absolute values
    of all Tensors in to_mask combined.

    The sparsity of masks already present in in_ckpt is taken into account -
    the current sparsity percentage is computed and subtracted from the desired
    sparsity percentage, the Tensors have their corresponding mask applied if
    present, and then a new threshold value is computed.

    Args:
        in_ckpt: Checkpoint to read Tensor values from.
        to_mask: Tensor names to include in threshold computation.
        sparsity: Sparsity percentage between 0.0 and 100.0 inclusive.
            (default 0.0)

    Returns:
        Dictionary of {name, threshold_value for name in to_mask}. Note that
        threshold_value will be the same for all entries in the dict as it is
        computed globally.
    '''
    tensor_values = np.array([])
    mask_values = np.array([])
    all_mask_names = [get_mask_name(name) for name in to_mask]
    for name, value in var_names_to_values.items():
        if name in to_mask:
            tensor_values = np.concatenate([tensor_values, value.flatten()])
        elif name in all_mask_names:
            mask_values = np.concatenate([mask_values, value.flatten()])
    k = tensor_threshold(tensor_values, sparsity, mask=mask_values)
    return {name: k for name in to_mask}

def tensor_threshold(tensor, sparsity, mask=None):
    '''Compute the threshold value for the given sparsity level.

    The sparsity of mask is taken into account if present.

    Args:
        tensor: Tensor to compute threshold for.
        sparsity: Sparsity percentage between 0.0 and 100.0 inclusive.
            (default 0.0)

    Returns:
        Threshold value >= 0.0.
    '''
    # The aim is to induce sparsity_diff more % sparsity in total. However we
    # are only able to induce sparsity in the remaining fraction of values thus
    # we scale it accordingly.
    old_sparsity = tensor_sparsity_percent(mask) if mask is not None else 0.0
    sparsity_diff = sparsity - old_sparsity
    if 0.0 > sparsity_diff > 100.0:
        raise ValueError('new sparsity (%.2f%%) must be >= current sparsity (%.2f%%) and <= 100.0' % (
            sparsity, old_sparsity))
    percentile = sparsity_diff / (1.0 - old_sparsity/100.0)
    percentile = min(100.0, percentile)

    values = np.abs(tensor.flatten())
    if mask is not None:
        values = values[mask.flatten() == 1]
    if values.size == 0:
        raise ValueError('no values found when computing threshold')

    return np.percentile(values, percentile)
