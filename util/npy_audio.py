from __future__ import absolute_import, print_function

import numpy as np
import sys

def audiofile_to_input_vector(audio_filename, numcep, numcontext):
    r"""
    Given an audio file at ``audio_filename``, calculates ``numcep`` MFCC features
    at every 0.01s time step with a window length of 0.025s. Appends ``numcontext``
    context frames to the left and right of each time step, and returns this data
    in a numpy array.
    """
    if numcep != 26 or numcontext != 9:
        raise ValueError('numcep must be 26 and numcontext must be 9!')

    # Load wav files
    features = np.load(audio_filename)
    num_strides = len(features)

    window_size = 2*numcontext+1
    train_inputs = np.lib.stride_tricks.as_strided(
        features,
        (num_strides, window_size, numcep),
        (features.strides[0], features.strides[0], features.strides[1]),
        writeable=False)

    # Flatten the second and third dimensions
    train_inputs = np.reshape(train_inputs, [num_strides, -1])
    train_inputs = train_inputs[:-2*numcontext, :]

    # Return results
    return train_inputs
