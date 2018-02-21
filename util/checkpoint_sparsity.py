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


def add_masks(out_ckpt, in_ckpt, to_mask, sparsity=0.0):
    '''Add masks for variables in to_mask to out_ckpt based on sparsity percent.

    Args:
        out_ckpt: Checkpoint to write Tensor and mask values to.
        in_ckpt: Checkpoint to read Tensor values from.
        to_mask: Tensor names to include in threshold computation.
        sparsity: Sparsity percentage between 0.0 and 100.0 inclusive. (default 0.0)
    '''
    k = threshold(in_ckpt, to_mask, sparsity)

    tf.reset_default_graph()
    reader = tf.train.NewCheckpointReader(in_ckpt)
    var_to_shape_map = reader.get_variable_to_shape_map()
    var_to_dtype_map = reader.get_variable_to_dtype_map()
    var_names_to_values = {}

    for name, shape in var_to_shape_map.items():
        # Create variable and add to value map.
        tensor = reader.get_tensor(name)
        dtype = var_to_dtype_map[name]
        tf.get_variable(name, shape=shape, dtype=dtype)
        var_names_to_values[name] = tensor

        if name not in to_mask:
            continue

        # Create mask and add to value map.
        mask_name = name + '/mask'
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
    of all tensors in to_mask combined.

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
    for name in var_to_shape_map:
        if name not in to_mask:
            continue

        tensor = reader.get_tensor(name)
        tensor = tensor.flatten()
        values = np.concatenate([values, tensor])

    values = np.abs(values)
    if values.size == 0:
        print('no values found when computing threshold')
        return 0
    return np.percentile(values, sparsity)


def validate_sparsity():
    if not (0.0 <= FLAGS.sparsity <= 100.0):
        print('sparsity must be between 0.0 and 100.0 inclusive, got %.2f' % FLAGS.sparsity)
        sys.exit(0)


if __name__ == '__main__':
    tf.app.run()
