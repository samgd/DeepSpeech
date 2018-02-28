import numpy as np
import os
import shutil
import tempfile
import tensorflow as tf
import unittest

import util.sparsity.mask as mask

from tensorflow.contrib.framework import assign_from_values

from util.sparsity.threshold import add_masks

class TestCheckpointSparsity(unittest.TestCase):

    def setUp(self):
        '''Create a temporary directory containing a checkpoint file.'''
        tf.reset_default_graph()

        # Create temporary directory and filename.
        self.test_dir = tempfile.mkdtemp()
        self.ckpt = os.path.join(self.test_dir, 'checkpoint.ckpt')

        # Initialize variables values.
        self.vars_to_shapes = {'foo': (100, 100), 'bar': (2500, 4)}
        self.vars_to_values = {name: np.random.randn(*shape)
                               for name, shape in self.vars_to_shapes.items()}
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
            sparsity = get_sparsity(tensor)
            self.assertAlmostEqual(sparsity, target_sparsity, places=1)


    def test_add_masks_creates_masks_correct_sparsity_percentage(self):
        '''Ensure mask added with correct sparsity percentage to checkpoint'''
        out_ckpt = self.ckpt + '-masks'
        target_sparsity = 40.0
        add_masks(out_ckpt,
                     self.ckpt,
                     to_mask=self.to_mask,
                     sparsity=target_sparsity)
        self.check_sparsity(out_ckpt, target_sparsity)


    def test_add_masks_increases_masks_sparsity_percentage(self):
        '''Ensure add_masks sparsity increases and masked values stay masked'''
        target_sparsity = [40.0, 45.5, 73.2, 99.1]
        in_ckpt = self.ckpt
        mask_log = []
        for i, sparsity in enumerate(target_sparsity):
            out_ckpt = in_ckpt + ('-%d' % i)
            add_masks(out_ckpt,
                         in_ckpt,
                         to_mask=self.to_mask,
                         sparsity=sparsity)
            in_ckpt = out_ckpt
            self.check_sparsity(out_ckpt, sparsity)
            # Log masks to test afterwards.
            mask_log.append(self.get_masks(out_ckpt))

        self.check_mask_subset(mask_log)


    def check_mask_subset(self, mask_log):
        '''Ensure each list of masks contains the previous as a subset.'''
        if len(mask_log) == 1:
            return
        pair_log = zip(mask_log, mask_log[1:])
        for prev_log, next_log in pair_log:
            # Match each mask in prev with its value in next.
            matching = zip(prev_log, next_log)
            for prev_mask, next_mask in matching:
                self.assertTrue(np.all(prev_mask >= next_mask))


    def get_masks(self, ckpt):
        '''Return a list of masks in ckpt'''
        reader = tf.train.NewCheckpointReader(ckpt)
        masks = []
        for name in self.to_mask:
            mask_name = mask.get_mask_name(name)
            masks.append(reader.get_tensor(mask_name))
        return masks


    def check_sparsity(self, ckpt, target_sparsity):
        '''Ensure masks mask off approx target_sparsity values'''
        reader = tf.train.NewCheckpointReader(ckpt)
        for name in self.vars_to_shapes:
            tensor = reader.get_tensor(name)
            self.assertTrue(np.allclose(tensor, self.vars_to_values[name]))

            if name not in self.to_mask:
                continue

            mask = reader.get_tensor(get_mask_name(name))
            sparsity = get_sparsity(mask)
            self.assertAlmostEqual(sparsity, target_sparsity, places=1)


def get_sparsity(tensor):
    return (float(np.sum(tensor == 0)) / tensor.size) * 100

def get_mask_name(name):
    return name + '/mask'


if __name__ == '__main__':
    unittest.main()
