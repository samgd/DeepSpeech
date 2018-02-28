import tensorflow as tf
import sys

import util.sparsity.threshold as threshold

tf.app.flags.DEFINE_string  ('in_ckpt',             '',    'checkpoint to read Tensor values from')
tf.app.flags.DEFINE_string  ('out_ckpt',            '',    'checkpoint to write Tensor and mask values to')
tf.app.flags.DEFINE_string  ('to_mask',             '',    'file containing names of Tensors to mask, one per line')
tf.app.flags.DEFINE_float   ('sparsity',            0.0,   'sparsity percentage between 0.0 and 100.0 inclusive.')
tf.app.flags.DEFINE_boolean ('layerwise_limit',     False, 'compute thresholds per layer if True and sparsify each layer up to sparsity percentage at most.')
tf.app.flags.DEFINE_string  ('opaque_params_name', 'fp32_storage/cudnn_lstm/opaque_kernel', 'name of opaque parameters, set to \'\' to avoid splitting opaque_params blob.')

FLAGS = tf.app.flags.FLAGS

def main(_):
    validate_sparsity()

    with open(FLAGS.to_mask, 'r') as to_mask_file:
        to_mask = to_mask_file.read().split('\n')
        # Remove empty strings.
        to_mask = [name for name in to_mask if name]

    if FLAGS.layerwise_limit:
        threshold.layerwise_add_masks_limit(FLAGS.out_ckpt,
                                            FLAGS.in_ckpt,
                                            to_mask,
                                            FLAGS.sparsity,
                                            FLAGS.opaque_params_name)
    else:
        threshold.add_masks(FLAGS.out_ckpt,
                            FLAGS.in_ckpt,
                            to_mask,
                            FLAGS.sparsity,
                            FLAGS.opaque_params_name)

def validate_sparsity():
    if not 0.0 <= FLAGS.sparsity <= 100.0:
        print('sparsity must be between 0.0 and 100.0 inclusive, got %.2f' % FLAGS.sparsity)
        sys.exit(0)

if __name__ == '__main__':
    tf.app.run()
