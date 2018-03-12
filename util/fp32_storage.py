import numpy as np
import tensorflow as tf

from tensorflow.contrib.framework import assign_from_values

NON_TRAINABLE_VARIABLES = ['global_step',
                           'beta2_power',
                           'beta1_power']

tf.app.flags.DEFINE_string ('in_ckpt',  '',              'checkpoint to read Tensor values from')
tf.app.flags.DEFINE_string ('out_ckpt', '',              'checkpoint to write Tensor values to')
tf.app.flags.DEFINE_string ('prefix',   'fp32_storage/', 'prefix to add to trainable variables')

FLAGS = tf.app.flags.FLAGS

def main(_):
    prefix_variables(out_ckpt=FLAGS.out_ckpt,
                     in_ckpt=FLAGS.in_ckpt,
                     prefix=FLAGS.prefix,
                     ignore=NON_TRAINABLE_VARIABLES)

def prefix_variables(out_ckpt, in_ckpt, prefix, ignore=None):
    tf.reset_default_graph()

    reader = tf.train.NewCheckpointReader(in_ckpt)
    shape_map = reader.get_variable_to_shape_map()
    dtype_map = reader.get_variable_to_dtype_map()

    var_names_to_values = {}
    for name, shape in shape_map.items():
        value = reader.get_tensor(name)
        dtype = dtype_map[name]

        if name not in ignore:
            name = prefix + name
        var_names_to_values[name] = value

        tf.get_variable(name, shape=shape, dtype=dtype)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        assign_op, feed_dict = assign_from_values(var_names_to_values)
        sess.run(assign_op, feed_dict)
        saver.save(sess, out_ckpt)

if __name__ == '__main__':
    tf.app.run()
