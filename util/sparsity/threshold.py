import numpy as np
import tensorflow as tf
import sys

from tensorflow.contrib.framework import assign_from_values

from util.sparsity.mask import get_mask_name
from util.sparsity.mask import tensor_sparsity_percent

def add_masks(out_ckpt, in_ckpt, to_mask, sparsity=0.0, use_layerwise_threshold=False):
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
    tf.reset_default_graph()

    if not use_layerwise_threshold:
        thresholds = global_threshold(in_ckpt, to_mask, sparsity)
    else:
        thresholds = layerwise_threshold(in_ckpt, to_mask, sparsity)

    reader = tf.train.NewCheckpointReader(in_ckpt)
    var_to_shape_map = reader.get_variable_to_shape_map()
    var_to_dtype_map = reader.get_variable_to_dtype_map()
    all_masks = [get_mask_name(name) for name in to_mask]
    var_names_to_values = {}

    for name, shape in var_to_shape_map.items():
        if name in all_masks:
            # Mask will be created later on.
            continue

        # Create variable in out_ckpt and add to value map.
        tensor = reader.get_tensor(name)
        dtype = var_to_dtype_map[name]
        tf.get_variable(name, shape=shape, dtype=dtype)
        var_names_to_values[name] = tensor

        if name not in to_mask:
            continue

        # Create mask in out_ckpt, ensuring to read mask in in_ckpt if it
        # exists and add to value map.
        mask_name = get_mask_name(name)
        if mask_name in var_to_shape_map:
            mask = reader.get_tensor(mask_name)
            tensor = np.multiply(tensor, mask)

        mask = np.abs(tensor) > thresholds[name]
        tf.get_variable(mask_name, shape=shape, dtype=dtype)
        var_names_to_values[mask_name] = mask

    saver = tf.train.Saver()
    with tf.Session() as sess:
        assign_op, feed_dict = assign_from_values(var_names_to_values)
        sess.run(assign_op, feed_dict)
        saver.save(sess, out_ckpt)

def layerwise_threshold(in_ckpt, to_mask, sparsity=0.0):
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
    reader = tf.train.NewCheckpointReader(in_ckpt)
    var_to_shape_map = reader.get_variable_to_shape_map()

    thresholds = {}
    for name in to_mask:
        tensor = reader.get_tensor(name)

        mask_name = get_mask_name(name)
        mask = reader.get_tensor(mask_name) if mask_name in var_to_shape_map else None
        thresholds[name] = tensor_threshold(tensor,
                                            sparsity,
                                            mask=mask)
    return thresholds

def global_threshold(in_ckpt, to_mask, sparsity=0.0):
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
    reader = tf.train.NewCheckpointReader(in_ckpt)
    var_to_shape_map = reader.get_variable_to_shape_map()

    tensor_values = np.array([])
    mask_values = np.array([])
    for name in to_mask:
        tensor = reader.get_tensor(name)
        tensor_values = np.concatenate([tensor_values, tensor.flatten()])

        mask_name = get_mask_name(name)
        if mask_name in var_to_shape_map:
            mask = reader.get_tensor(mask_name)
        else:
            mask = np.ones(tensor.shape)
        mask_values = np.concatenate([mask_values, mask.flatten()])

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

    values = np.abs(tensor.flatten())
    if mask is not None:
        values = values[mask.flatten() == 1]
    if values.size == 0:
        raise ValueError('no values found when computing threshold')

    return np.percentile(values, percentile)
