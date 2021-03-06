import numpy as np
import tensorflow as tf
import sys

from tensorflow.contrib.framework import assign_from_values

from util import convert_params
from util.sparsity.mask import get_mask_name
from util.sparsity.mask import tensor_sparsity_percent

def layerwise_add_masks_limit(out_ckpt, in_ckpt, to_mask, limit_sparsity=0.0,
                              cudnn_params_name=''):
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
              cudnn_params_name=cudnn_params_name)

def add_masks(out_ckpt, in_ckpt, to_mask, sparsity=0.0,
              use_layerwise_threshold=False,
              cudnn_params_name=''):
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
    if use_layerwise_threshold and cudnn_params_name:
        var_names_to_values = get_and_split_cudnn(in_ckpt, cudnn_params_name)
        ignore_params = [cudnn_params_name, get_mask_name(cudnn_params_name)]
    else:
        var_names_to_values = {}
        ignore_params = []
    var_names_to_values = get_values(in_ckpt, to_mask, var_names_to_values, ignore_params)
    if not use_layerwise_threshold:
        thresholds = global_threshold(var_names_to_values, to_mask, sparsity)
    else:
        thresholds = layerwise_threshold(var_names_to_values, to_mask, sparsity)
    var_names_to_values = update_masks(var_names_to_values, to_mask, thresholds)
    if use_layerwise_threshold and cudnn_params_name:
        var_names_to_values = join_cudnn(var_names_to_values, cudnn_params_name)
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
        dtype = tf.int32 if name == 'global_step' else tf.float32
        tf.get_variable(name, shape=value.shape, dtype=dtype)

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

def get_values(ckpt, to_mask, extra_params=None, ignore_params=None):
    '''Return a dict of Tensor names in to_mask, and mask names, to values.'''
    reader = tf.train.NewCheckpointReader(ckpt)
    var_to_shape_map = reader.get_variable_to_shape_map()

    var_names_to_values = {}
    all_names = var_to_shape_map.keys()
    if extra_params:
        all_names.extend(extra_params.keys())

    for name in all_names:
        if ignore_params and name in ignore_params:
            continue
        # Get Tensor and get or create mask.
        if name in var_to_shape_map:
            tensor = reader.get_tensor(name)
        elif name in extra_params:
            tensor = extra_params[name]
        else:
            raise ValueError('unable to get %r' % name)

        var_names_to_values[name] = tensor

        if name not in to_mask:
            continue

        mask_name = get_mask_name(name)
        if mask_name in var_to_shape_map:
            mask = reader.get_tensor(mask_name)
        elif mask_name in extra_params:
            mask = extra_params[mask_name]
        else:
            mask = np.ones(tensor.shape)
        # Store values to manipulate later.
        var_names_to_values[mask_name] = mask

    return var_names_to_values

def get_and_split_cudnn(ckpt, cudnn_params_name):
    var_names_to_values = get_values(ckpt, [cudnn_params_name])
    updated_values = {}

    params = var_names_to_values[cudnn_params_name]
    updated_values.update(convert_params.cudnn_to_canonical(params))

    mask = var_names_to_values[get_mask_name(cudnn_params_name)]
    split_mask = convert_params.cudnn_to_canonical(mask)
    split_mask = {get_mask_name(name): val for name, val in split_mask.items()}
    updated_values.update(split_mask)

    for name, value in var_names_to_values.items():
        if name == cudnn_params_name:
            continue
        updated_values[name] = value

    return updated_values

def join_cudnn(var_names_to_values, cudnn_params_name):
    updated_values = {}
    # Reconstruct parameter blob.
    weight_shapes = convert_params.get_weight_shapes('fw_')
    weight_shapes.extend(convert_params.get_weight_shapes('bw_'))
    names = [name for name, _ in weight_shapes]
    names.extend(convert_params.get_bias_names('fw_'))
    names.extend(convert_params.get_bias_names('bw_'))
    split_params = {name: var_names_to_values[name] for name in names}
    updated_values[cudnn_params_name] = convert_params.cudnn_to_canonical(split_params)
    # Reconstruct parameter blob mask.
    split_mask = {name: var_names_to_values[get_mask_name(name)] for name in names}
    mask = convert_params.cudnn_to_canonical(split_mask)
    updated_values[get_mask_name(cudnn_params_name)] = mask
    # Add remaining variables to new dict.
    all_names = names + [get_mask_name(name) for name in names]
    for name, value in var_names_to_values.items():
        if name in all_names:
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
    if sparsity_diff < 0 or sparsity_diff > 100.0:
        return 0.0
    norm = 1.0 - old_sparsity/100.0
    if norm == 0.0:
        norm = 1.0
    percentile = sparsity_diff / norm
    percentile = min(100.0, percentile)

    values = np.abs(tensor.flatten())
    if mask is not None:
        values = values[mask.flatten() == 1]
    if values.size == 0:
        return 0.0

    return np.percentile(values, percentile)
