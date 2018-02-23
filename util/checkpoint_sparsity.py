import numpy as np
import tensorflow as tf
import sys

from tensorflow.contrib.framework import assign_from_values

tf.app.flags.DEFINE_string  ('in_ckpt',  '',  'checkpoint to read Tensor values from')
tf.app.flags.DEFINE_string  ('out_ckpt', '',  'checkpoint to write Tensor and mask values to')
tf.app.flags.DEFINE_string  ('to_mask',  '',  'file containing names of Tensors to mask, one per line')
tf.app.flags.DEFINE_float   ('sparsity', 0.0, 'sparsity percentage between 0.0 and 100.0 inclusive.')

FLAGS = tf.app.flags.FLAGS

def main(_):
    validate_sparsity()

    with open(FLAGS.to_mask, 'r') as f:
        to_mask = f.read().split('\n')

    add_masks(FLAGS.out_ckpt,
              FLAGS.in_ckpt,
              to_mask,
              FLAGS.sparsity)

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


def add_masks(out_ckpt, in_ckpt, to_mask, sparsity=0.0):
    '''Add masks for variables in to_mask to out_ckpt based on sparsity percent.

    The sparsity of masks already present in in_ckpt is taken into account -
    the current sparsity percentage is computed and subtracted from the desired
    sparsity percentage. A new mask is then computed whilst ensuring the
    current mask is preserved if present.

    Args:
        out_ckpt: Checkpoint to write Tensor and mask values to.
        in_ckpt: Checkpoint to read Tensor values from.
        to_mask: Tensor names to include in threshold computation.
        sparsity: Sparsity percentage between 0.0 and 100.0 inclusive. (default 0.0)
    '''
    tf.reset_default_graph()

    k = threshold(in_ckpt, to_mask, sparsity)

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
        mask = np.zeros(shape)
        if mask_name in var_to_shape_map:
            mask = reader.get_tensor(mask_name)
            tensor = np.multiply(tensor, mask)

        mask = np.abs(tensor) > k
        tf.get_variable(mask_name, shape=shape, dtype=dtype)
        var_names_to_values[mask_name] = mask

    saver = tf.train.Saver()
    with tf.Session() as sess:
        assign_op, feed_dict = assign_from_values(var_names_to_values)
        sess.run(assign_op, feed_dict)
        saver.save(sess, out_ckpt)


def threshold(in_ckpt, to_mask, sparsity=0.0):
    '''Compute the threshold value for the given sparsity level.

    The threshold value is the `sparsity'th percentile of the absolute values
    of all Tensors in to_mask combined.

    The sparsity of masks already present in in_ckpt is taken into account -
    the current sparsity percentage is computed and subtracted from the desired
    sparsity percentage, the Tensors have their corresponding mask applied if
    present, and then a new threshold value is computed.

    Args:
        in_ckpt: Checkpoint to read Tensor values from.
        to_mask: Tensor names to include in threshold computation.
        sparsity: Sparsity percentage between 0.0 and 100.0 inclusive. (default 0.0)

    Returns:
        Threshold value >= 0.0.
    '''
    reader = tf.train.NewCheckpointReader(in_ckpt)
    var_to_shape_map = reader.get_variable_to_shape_map()

    values = np.array([])
    total_params = 0
    total_masked = 0
    for name in var_to_shape_map:
        if name not in to_mask:
            continue

        tensor = reader.get_tensor(name)
        total_params += tensor.size

        mask_name = get_mask_name(name)
        if mask_name in var_to_shape_map:
            mask = reader.get_tensor(mask_name)
            total_masked += np.sum(mask == 0)
            tensor = tensor[mask == 1]

        tensor = tensor.flatten()
        values = np.concatenate([values, tensor])

    values = np.abs(values)
    if values.size == 0:
        print('no values found when computing threshold')
        return 0

    old_sparsity = (float(total_masked) / total_params) * 100.0
    sparsity_diff = sparsity - old_sparsity
    if sparsity_diff < 0:
        raise ValueError('new sparsity (%.2f%%) must be >= current sparsity (%.2f%%)' % (sparsity, old_sparsity))
    # The aim is to induce sparsity_diff more % sparsity in total. However we
    # are only able to induce sparsity in the remaining fraction of values thus
    # we scale it accordingly.
    percentile = (float(total_params) / values.size) * sparsity_diff
    percentile = min(percentile, 100.0) # Fix rounding errors.

    return np.percentile(values, percentile)


def get_mask_name(tensor_name):
    '''Return the name of the mask for a given a Tensor.'''
    return tensor_name + '/mask'


def validate_sparsity():
    if not (0.0 <= FLAGS.sparsity <= 100.0):
        print('sparsity must be between 0.0 and 100.0 inclusive, got %.2f' % FLAGS.sparsity)
        sys.exit(0)


if __name__ == '__main__':
    tf.app.run()
