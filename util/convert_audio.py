import csv
import numpy as np
import os
import soundfile as sf
import tensorflow as tf

tf.app.flags.DEFINE_string('wav_csv', '', '')

def wav_to_flac(wav_csv):
    root, _ = os.path.splitext(wav_csv)

    flac_csv = root + '-flac.csv'
    with open(flac_csv, 'w') as flac_csv_file:
        writer = csv.writer(flac_csv_file, delimiter=',')
        writer.writerow(['flac_filename', 'flac_filesize', 'transcript'])

        with open(wav_csv, 'r') as wav_csv_file:
            reader = csv.reader(wav_csv_file, delimiter=',')
            reader.next()
            for i, row in enumerate(reader):
                # Load wav file.
                wav_filename = row[0]
                audio, fs = sf.read(wav_filename, dtype=np.int16)

                # Convert to flac and write.
                audio_root, _ = os.path.splitext(wav_filename)
                flac_filename = audio_root + '.flac'
                sf.write(flac_filename, audio, fs)

                # Get filesize.
                filesize = os.path.getsize(flac_filename)
                writer.writerow([flac_filename, filesize, row[2]])

                if i % 50 == 0:
                    flac_csv_file.flush()

if __name__ == '__main__':
    wav_to_flac(tf.app.flags.FLAGS.wav_csv)
