import numpy as np
import os
import shutil
import tempfile
import tensorflow as tf
import unittest

import util.sparsity.mask as mask

from tensorflow.contrib.framework import assign_from_values

from util.sparsity.threshold import add_masks

class TestMask(unittest.TestCase):

    def setUp(self):
        '''Create a temporary directory containing a checkpoint file.'''
        tf.reset_default_graph()

        # Create temporary directory and filename.
        self.test_dir = tempfile.mkdtemp()
        self.ckpt = os.path.join(self.test_dir, 'checkpoint.ckpt')

        # Initialize variables values.
        self.var_names = ['foo', 'bar']
        self.vars_to_values = {name: gen_rnd_tensor()
                               for name in self.var_names}
        self.vars_to_shapes = {name: val.shape
                               for name, val in self.vars_to_values.items()}
        self.to_mask = ['foo']

        # Create TensorFlow graph containing the variables.
        for name, value in self.vars_to_values.items():
            shape = self.vars_to_shapes[name]
            tf.get_variable(name, shape=shape)

        # Save to checkpoint.
        saver = tf.train.Saver()
        with tf.Session() as sess:
            assign_op, feed_dict = assign_from_values(self.vars_to_values)
            sess.run(assign_op, feed_dict)
            saver.save(sess, self.ckpt)

    def tearDown(self):
        '''Clean-up: Remove temporary checkpoint directory.'''
        shutil.rmtree(self.test_dir)

    def test_tensor_sparsity_percent(self):
        '''Ensure correct sparsity percentage returned.'''
        for name, tensor in self.vars_to_values.items():
            abs_tensor = np.abs(tensor)
            target = np.random.uniform(0, 100)
            m = abs_tensor > np.percentile(abs_tensor, target)
            self.assertAlmostEqual(mask.tensor_sparsity_percent(m),
                                   target,
                                   places=2)

    def test_apply_masks(self):
        # Create sparsity masks.
        mask_ckpt = self.ckpt + '-masks'
        target_sparsity = 40.0
        add_masks(mask_ckpt,
                  self.ckpt,
                  to_mask=self.to_mask,
                  sparsity=target_sparsity)
        app_ckpt = self.ckpt + '-app'

        # Apply sparsity masks.
        mask.apply_masks(app_ckpt, mask_ckpt)
        reader = tf.train.NewCheckpointReader(app_ckpt)
        for name in self.to_mask:
            tensor = reader.get_tensor(name)
            sparsity = mask.tensor_sparsity_percent(tensor)
            self.assertAlmostEqual(sparsity, target_sparsity, places=1)

def gen_rnd_tensor():
    rnd_shape = np.random.uniform(500, 1000, size=(2,)).astype(np.int)
    sigm = np.random.uniform(0.01, 2)
    mean = np.random.uniform(-2, 2)
    return sigm * np.random.randn(np.prod(rnd_shape)).reshape(rnd_shape) + mean
