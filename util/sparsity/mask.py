import numpy as np
import tensorflow as tf

from tensorflow.contrib.framework import assign_from_values


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
