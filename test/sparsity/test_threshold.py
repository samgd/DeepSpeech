import numpy as np
import os
import shutil
import tempfile
import tensorflow as tf
import unittest

import util.sparsity.mask as mask

from tensorflow.contrib.framework import assign_from_values

from util.sparsity.threshold import add_masks

class TestThreshold(unittest.TestCase):

    def setUp(self):
        '''Create a temporary directory containing a checkpoint file.'''
        tf.reset_default_graph()

        # Create temporary directory and filename.
        self.test_dir = tempfile.mkdtemp()
        self.ckpt = os.path.join(self.test_dir, 'checkpoint.ckpt')

        # Initialize variables values.
        self.vars_to_shapes = {'foo': (100, 100),
                               'bar': (2500, 4),
                               'baz': (3000, 100)}
        mean = lambda: np.random.uniform(-30, 30)
        std = lambda: np.random.uniform(0.1, 4)
        self.vars_to_values = {name: std() * np.random.randn(*shape) + mean()
                               for name, shape in self.vars_to_shapes.items()}
        self.to_mask = ['foo', 'baz']

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

    def test_add_masks_globally_creates_masks_correct_ckpt_sparsity_percentage(self):
        '''Ensure masks added to checkpoint and total sparsity is correct.'''
        out_ckpt = self.ckpt + '-masks'
        target_sparsity = 40.0
        add_masks(out_ckpt,
                  self.ckpt,
                  to_mask=self.to_mask,
                  sparsity=target_sparsity)
        self.check_sparsity(out_ckpt, target_sparsity, global_=True)

    def test_add_masks_layerwise_creates_masks_equal_sparsity_percentage(self):
        '''Ensure masks added to checkpoint with equal sparsity.'''
        out_ckpt = self.ckpt + '-masks'
        target_sparsity = 66.0
        add_masks(out_ckpt,
                  self.ckpt,
                  to_mask=self.to_mask,
                  sparsity=target_sparsity,
                  use_layerwise_threshold=True)
        self.check_sparsity(out_ckpt, target_sparsity, global_=False)

    def test_add_masks_globally_increases_ckpt_sparsity_percentage(self):
        '''Ensure add_masks increases total sparsity in checkpoint.'''
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
            self.check_sparsity(out_ckpt, sparsity, global_=True)
            # Log masks to test afterwards.
            mask_log.append(self.get_masks(out_ckpt))
        self.check_mask_subset(mask_log)

    def test_add_masks_layerwise_increases_sparsity_percentage(self):
        '''Ensure add_masks increases total sparsity in checkpoint.'''
        target_sparsity = [40.0, 45.5, 73.2, 99.1]
        in_ckpt = self.ckpt
        mask_log = []
        for i, sparsity in enumerate(target_sparsity):
            out_ckpt = in_ckpt + ('-%d' % i)
            add_masks(out_ckpt,
                      in_ckpt,
                      to_mask=self.to_mask,
                      sparsity=sparsity,
                      use_layerwise_threshold=True)
            in_ckpt = out_ckpt
            self.check_sparsity(out_ckpt, sparsity, global_=False)
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

    def check_sparsity(self, ckpt, target_sparsity, global_=True):
        '''Ensure masks mask off approx target_sparsity values'''
        if global_:
            self.assertAlmostEqual(mask.ckpt_sparsity_percent(ckpt, self.to_mask),
                                   target_sparsity,
                                   places=2)
        else:
            reader = tf.train.NewCheckpointReader(ckpt)
            for name in self.vars_to_shapes:
                tensor = reader.get_tensor(name)
                self.assertTrue(np.allclose(tensor, self.vars_to_values[name]))

                if name not in self.to_mask:
                    continue

                mask_values = reader.get_tensor(mask.get_mask_name(name))
                sparsity = mask.tensor_sparsity_percent(mask_values)
                self.assertAlmostEqual(sparsity, target_sparsity, places=1)
