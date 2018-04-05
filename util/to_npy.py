import csv
import errno
import numpy as np
import os
import random
import scipy.io.wavfile as wav
import sox

from audio import audiofile_to_input_vector
from python_speech_features import mfcc


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def shrink(audio_filename, numcep=26, numcontext=9):
    vec = audiofile_to_input_vector(audio_filename, numcep, numcontext)

    mean_wrap = np.full((numcontext, numcep), vec[0, 0], dtype=vec.dtype)
    no_context = vec[:, 9*numcep:10*numcep]
    no_context = np.concatenate((mean_wrap, no_context, mean_wrap))
    features = no_context
    features = features.astype(np.float16)
    return features

in_dir = "/home/ubuntu/Code/DeepSpeech/data/librivox/"
out_dir = "/home/ubuntu/npy_data/"
tmp_file = "/tmp/data_aug.wav"

csv_files = ['librivox-train-clean-100.csv',
             'librivox-train-clean-360.csv',
             'librivox-train-other-500.csv']

tempo_bounds = [0.8, 1.2]
pitch_bounds = [-1, 1]


for csv_file in csv_files:
    csv_in_file = os.path.join(in_dir, csv_file)
    csv_out_file = os.path.join(out_dir, csv_file)

    numcontext = 9
    numcep = 26

    with open(csv_out_file, 'w') as out_handle:
        writer = csv.writer(out_handle, delimiter=',', lineterminator="\n")

        with open(csv_in_file) as csv_handle:
            csv_data = csv.reader(csv_handle, delimiter=',')
            writer.writerow(csv_data.next())

            for i, row in enumerate(csv_data):
                tfm = sox.Transformer()
                tfm.tempo(random.uniform(*tempo_bounds))
                tfm.pitch(random.uniform(*pitch_bounds))
                tfm.build(row[0], tmp_file)

                print(row)
                features = shrink(tmp_file)

                base, _ = os.path.splitext(row[0])
                base = base[len(in_dir):]
                base = os.path.join(out_dir, base)
                mkdir_p(os.path.dirname(base))
                np.save(base, features)
                writer.writerow([str(i), base + '.npy', row[1], row[2]])
