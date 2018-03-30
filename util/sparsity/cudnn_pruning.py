import numpy as np
import tensorflow as tf

from tensorflow.contrib.cudnn_rnn.python.layers import cudnn_rnn
from tensorflow.contrib.model_pruning.python import pruning
from tensorflow.contrib.model_pruning import get_pruning_hparams
from tensorflow.contrib.model_pruning.python.pruning import apply_mask

_CUDNN_MASK_COLLECTION = 'cudnn_masks'
_CUDNN_THRESHOLD_COLLECTION = 'cudnn_thresholds'
_CUDNN_MASKED_WEIGHT_COLLECTION = 'cudnn_masked_weights'
_CUDNN_WEIGHT_COLLECTION = 'cudnn_kernel'
_CUDNN_MASKED_WEIGHT_NAME = 'weights/cudnn_masked_weights'

_CUDNN_PARAMS_RNN_MODE   = 'cudnn_params_rnn_mode'
_CUDNN_PARAMS_NUM_LAYERS = 'cudnn_params_num_layers'
_CUDNN_PARAMS_NUM_UNITS  = 'cudnn_params_num_units'
_CUDNN_PARAMS_SHAPE      = 'cudnn_params_shape'
_CUDNN_PARAMS_MODE       = 'cudnn_params_mode'
_CUDNN_PARAMS_DIR        = 'cudnn_params_dir'
_CUDNN_PARAMS_DO         = 'cudnn_params_do'
_CUDNN_PARAMS_SEED       = 'cudnn_params_seed'

_CUDNN_PARAMS = [_CUDNN_PARAMS_RNN_MODE,
                 _CUDNN_PARAMS_NUM_LAYERS,
                 _CUDNN_PARAMS_NUM_UNITS,
                 _CUDNN_PARAMS_SHAPE,
                 _CUDNN_PARAMS_MODE,
                 _CUDNN_PARAMS_DIR,
                 _CUDNN_PARAMS_DO,
                 _CUDNN_PARAMS_SEED]

class MaskedCudnnLSTM(cudnn_rnn.CudnnLSTM):

    def __init__(self,
                 num_layers,
                 num_units,
                 input_mode=cudnn_rnn.CUDNN_INPUT_LINEAR_MODE,
                 direction=cudnn_rnn.CUDNN_RNN_UNIDIRECTION,
                 dropout=0.,
                 seed=None,
                 dtype=tf.float32,
                 kernel_initializer=None,
                 bias_initializer=None,
                 name=None):
        super(MaskedCudnnLSTM, self).__init__(num_layers,
                                              num_units,
                                              input_mode,
                                              direction,
                                              dropout,
                                              seed,
                                              dtype,
                                              kernel_initializer,
                                              bias_initializer,
                                              name)

    def build(self, inputs_shape):
        super(MaskedCudnnLSTM, self).build(inputs_shape)

        self.built = False

        # Create masks and thresholds for all canonical weights.
        self._masks = []
        self._thresholds = []
        for i, shape in enumerate(self.canonical_weight_shapes):
            mask = self.add_variable(
                name='mask_%d' % i,
                shape=shape,
                initializer=tf.ones_initializer(),
                trainable=False,
                dtype=self.dtype)
            self._masks.append(mask)

            threshold = self.add_variable(
                name='threshold_%d' % i,
                shape=[],
                initializer=tf.zeros_initializer(),
                trainable=False,
                dtype=tf.float32)
            self._thresholds.append(threshold)

        passthru_biases = [tf.ones(shape, self.dtype)
                           for shape in self.canonical_bias_shapes]

        self._mask = self._canonical_to_opaque(self._masks, passthru_biases)
        self._masked_kernel = tf.multiply(self._mask, self.kernel,
                                          _CUDNN_MASKED_WEIGHT_NAME)

        if self._masked_kernel not in tf.get_collection_ref(_CUDNN_MASKED_WEIGHT_COLLECTION):
            tf.add_to_collection(_CUDNN_MASKED_WEIGHT_COLLECTION, self._masked_kernel)

            for mask in self._masks:
                tf.add_to_collection(_CUDNN_MASK_COLLECTION, mask)

            for threshold in self._thresholds:
                tf.add_to_collection(_CUDNN_THRESHOLD_COLLECTION, threshold)

            tf.add_to_collection(_CUDNN_WEIGHT_COLLECTION, self.kernel)

            params = [self._rnn_mode,
                      self._num_layers,
                      self._num_units,
                      inputs_shape[-1].value,
                      self._input_mode,
                      self._direction,
                      self._dropout,
                      self._seed]
            for collection, param in zip(_CUDNN_PARAMS, params):
                tf.add_to_collection(collection, param)

        self.built = True

    def call(self, inputs, initial_state=None, training=True):
        """Runs the forward step for the RNN model."""
        if initial_state is not None and not isinstance(initial_state, tuple):
            raise ValueError("Invalid initial_state type: %s, expecting tuple.",
                    type(initial_state))
        dtype = self.dtype
        inputs = tf.convert_to_tensor(inputs, dtype=dtype)

        batch_size = tf.shape(inputs)[1]
        if initial_state is None:
            initial_state = self._zero_state(batch_size)
        if self._rnn_mode == cudnn_rnn.CUDNN_LSTM:
            h, c = initial_state
        else:
            h, = initial_state
        h = tf.convert_to_tensor(h, dtype=dtype)
        if self._rnn_mode == cudnn_rnn.CUDNN_LSTM:
            c = tf.convert_to_tensor(c, dtype=dtype)
        else:
            # For model that doesn't take input_c, replace with a dummy tensor.
            c = tf.constant([], dtype=dtype)
        outputs, (output_h, output_c) = self._forward(inputs,
                                                      h, c,
                                                      self._masked_kernel,
                                                      training)

        if self._rnn_mode == cudnn_rnn.CUDNN_LSTM:
            return outputs, (output_h, output_c)
        else:
            return outputs, (output_h,)

def get_weight_sparsity():
    n_zero = []
    n_total = []
    for t in get_masks():
        n_zero.append(tf.reduce_sum(tf.cast(tf.equal(t, 0.0), tf.float32)))
        n_total.append(tf.cast(tf.size(t), tf.float32))
    return tf.add_n(n_zero) / tf.add_n(n_total)

def get_masks():
    masks = pruning.get_masks()
    cudnn_masks = get_cudnn_masks()
    return masks + cudnn_masks

def get_cudnn_masked_weights():
    return tf.get_collection(_CUDNN_MASKED_WEIGHT_COLLECTION)

def get_cudnn_masks():
    return tf.get_collection(_CUDNN_MASK_COLLECTION)

def get_cudnn_thresholds():
    return tf.get_collection(_CUDNN_THRESHOLD_COLLECTION)

def get_cudnn_weights():
    return tf.get_collection(_CUDNN_WEIGHT_COLLECTION)

def get_cudnn_params():
    collections = [tf.get_collection(collection) for collection in _CUDNN_PARAMS]
    return zip(*collections)

class CudnnPruning(pruning.Pruning):

    def __init__(self, spec=None, global_step=None, sparsity=None):
        super(CudnnPruning, self).__init__(spec=spec,
                                           global_step=global_step,
                                           sparsity=sparsity)

    def _get_mask_assign_ops(self):
        super(CudnnPruning, self)._get_mask_assign_ops()

        params = get_cudnn_params()
        weights = get_cudnn_weights()
        masks = get_cudnn_masks()
        thresholds = get_cudnn_thresholds()

        for param, cudnn_weight in zip(params, weights):
            canonical_weights, _ = cudnn_rnn.cudnn_rnn_ops.cudnn_rnn_opaque_params_to_canonical(
                rnn_mode=param[0],
                num_layers=param[1],
                num_units=param[2],
                input_size=param[3],
                params=cudnn_weight,
                input_mode=param[4],
                direction=param[5],
                dropout=param[6],
                seed=param[7])

            num_canon = len(canonical_weights)
            canonical_masks = masks[:num_canon]
            masks = masks[num_canon:]

            canonical_thresholds = thresholds[:num_canon]
            thresholds = thresholds[num_canon:]

            for i, mask in enumerate(canonical_masks):
                weight = canonical_weights[i]
                threshold = canonical_thresholds[i]

                # TODO: Partitioned?
                # TODO: do_not_prune?

                new_threshold, new_mask = self._update_mask(weight, threshold)
                self._assign_ops.append(pruning._variable_assign(threshold, new_threshold))
                self._assign_ops.append(pruning._variable_assign(mask, new_mask))

    def add_pruning_summaries(self):
        super(CudnnPruning, self).add_pruning_summaries()
        with tf.name_scope(self._spec.name + '_summaries'):
            masks = get_cudnn_masks()
            thresholds = get_cudnn_thresholds()
            for i, mask in enumerate(masks):
                tf.summary.scalar(mask.name + '/sparsity', tf.nn.zero_fraction(mask))
                tf.summary.scalar(thresholds[i].op.name + '/threshold', thresholds[i])
