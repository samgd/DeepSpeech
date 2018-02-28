import numpy as np
import tensorflow as tf
import sys

from tensorflow.contrib.framework import assign_from_values

from util.sparsity.threshold import add_masks

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


def validate_sparsity():
    if not 0.0 <= FLAGS.sparsity <= 100.0:
        print('sparsity must be between 0.0 and 100.0 inclusive, got %.2f' % FLAGS.sparsity)
        sys.exit(0)


if __name__ == '__main__':
    tf.app.run()
