import numpy as np
import os
import copy
import pandas
import random
import tensorflow as tf
import time

from math import ceil
from six.moves import range
from threading import Thread, Lock
from util.npy_audio import audiofile_to_input_vector
from util.gpu import get_available_gpus
from util.text import ctc_label_dense_to_sparse, text_to_char_array

class ModelFeeder(object):
    '''
    Feeds data into a model.
    Feeding is parallelized by independent units called tower feeders (usually one per GPU).
    Each tower feeder provides data from three runtime switchable sources (train, dev, test).
    These sources are to be provided by three DataSet instances whos references are kept.
    Creates, owns and delegates to tower_feeder_count internal tower feeder objects.
    '''
    def __init__(self,
                 train_set,
                 dev_set,
                 test_set,
                 numcep,
                 numcontext,
                 alphabet,
                 tower_feeder_count=-1,
                 threads_per_queue=1,
                 dtype=tf.float32,
                 logdir=''):

        self.train = train_set
        self.dev = dev_set
        self.test = test_set
        self.sets = [train_set, dev_set, test_set]
        self.numcep = numcep
        self.numcontext = numcontext
        self.tower_feeder_count = max(len(get_available_gpus()), 1) if tower_feeder_count < 0 else tower_feeder_count
        self.threads_per_queue = threads_per_queue

        self.ph_x = tf.placeholder(dtype, [None, None, numcep + (2 * numcep * numcontext)])
        self.ph_x_length = tf.placeholder(tf.int32, [None,])
        self.ph_y = tf.placeholder(tf.int32, [None,None,])
        self.ph_flat_y = tf.placeholder(tf.int32, [None,])
        self.ph_y_length = tf.placeholder(tf.int32, [None,])
        self.ph_queue_selector = tf.placeholder(tf.int32, name='Queue_Selector')

        self._tower_feeders = [_TowerFeeder(self, i, alphabet, dtype, logdir) for i in range(self.tower_feeder_count)]

    def start_queue_threads(self, session, coord):
        '''
        Starts required queue threads on all tower feeders.
        '''
        queue_threads = []
        for tower_feeder in self._tower_feeders:
            queue_threads += tower_feeder.start_queue_threads(session, coord)
        return queue_threads

    def empty_queues(self, session):
        for tower_feeder in self._tower_feeders:
            tower_feeder.empty_queues(session)

    def close_queues(self, session):
        '''
        Closes queues of all tower feeders.
        '''
        for tower_feeder in self._tower_feeders:
            tower_feeder.close_queues(session)

    def set_data_set(self, feed_dict, data_set):
        '''
        Switches all tower feeders to a different source DataSet.
        The provided feed_dict will get enriched with required placeholder/value pairs.
        The DataSet has to be one of those that got passed into the constructor.
        '''
        index = self.sets.index(data_set)
        assert index >= 0
        feed_dict[self.ph_queue_selector] = index

    def next_batch(self, tower_feeder_index):
        '''
        Draw the next batch from one of the tower feeders.
        '''
        return self._tower_feeders[tower_feeder_index].next_batch()

class DataSet(object):
    '''
    Represents a collection of audio samples and their respective transcriptions.
    Takes a set of CSV files produced by importers in /bin.

    next_index: Function to compute index of next batch. Note that the result
    is taken modulo the total number of batches.
    '''
    def __init__(self, name, csvs, target_batch_size, max_seq_len, skip=0, limit=0,
                 ascending=True, next_index=lambda i: i + 1,
                 shuffle_batch_order=True, shuffle_first_iteration=False, shuffle_seed=1234):

        self.name = name
        self.target_batch_size = target_batch_size
        self.max_seq_len = max_seq_len

        self.next_index = next_index

        self.files = None
        self.file_sets = []

        for set_csvs in csvs:
            files = None
            for csv in set_csvs:
                file = pandas.read_csv(csv, encoding='utf-8')
                if files is None:
                    files = file
                else:
                    files = files.append(file)
            files = files.sort_values(by="seq_len", ascending=ascending) \
                         .ix[:, ["wav_filename", "transcript", "seq_len"]] \
                         .values[skip:]
            if limit > 0:
                files = files[:limit]
            self.file_sets.append(files)

        self.batch_index_sets = self._create_batch_indices()
        self.total_batch_sets = [len(batch_indices) for batch_indices in self.batch_index_sets]

        self.current_set = -1
        self.current_batch = -1
        self.n_batch = 0

        self.next_set()

        self.shuffle_batch_order = shuffle_batch_order
        self.shuffle_seed = shuffle_seed
        if shuffle_batch_order and shuffle_first_iteration:
            random.seed(self.shuffle_seed)
            self.shuffle_seed += 3
            random.shuffle(self.batch_indices)

        self._lock = Lock()

    def next_set(self):
        self.current_set = (self.current_set + 1) % len(self.file_sets)
        self.files = self.file_sets[self.current_set]
        self.batch_indices = self.batch_index_sets[self.current_set]
        self.total_batches = self.total_batch_sets[self.current_set]

    def _create_batch_indices(self, multiple_of=8):
        '''Return a list of groups (lists) of batch indices into self.files.

        The sum of the sequence lengths in each batch is guaranteed to be less
        than or equal to target_batch_size * max_seq_len.

        Args:
            multiple_of: Each batch will be a multiple_of this number, except
                if there are too few elements.
        '''
        batch_index_sets = []
        for files in self.file_sets:
            batch_indices = []

            max_batch_values = self.target_batch_size * self.max_seq_len
            current_batch = []
            current_batch_lens = []
            current_batch_len = 0
            for i, row in enumerate(files):
                if current_batch_len + row[2] > max_batch_values:
                    # Ensure batch is a multiple of the desired number
                    split = (len(current_batch) // multiple_of) * multiple_of
                    batch_multiple_of = current_batch[:split]
                    if batch_multiple_of:
                        batch_indices.append(batch_multiple_of)
                    current_batch = current_batch[split:]
                    current_batch_lens = current_batch_lens[split:]
                    current_batch_len = sum(current_batch_lens)

                current_batch.append(i)
                current_batch_lens.append(row[2])
                current_batch_len += row[2]

            if current_batch:
                batch_indices.append(current_batch)

            batch_index_sets.append(batch_indices)

        return batch_index_sets

    @property
    def mean_batch_size(self):
        batch_lens = [len(batch) for batch in self.batch_indices]
        return float(sum(batch_lens)) / len(batch_lens)

    def next_batch_indices(self):
        with self._lock:
            idx = self.next_index(self.current_batch)
            if idx is None:
                return None

            if idx >= self.total_batches:
                if (not self.total_batches or
                        (self.n_batch % self.total_batches == 0
                         and self.shuffle_batch_order)):
                    random.seed(self.shuffle_seed)
                    self.shuffle_seed += 3
                    random.shuffle(self.batch_indices)
                return []
            self.n_batch += 1
            next_batch = self.batch_indices[idx]
            self.current_batch = idx
	    return next_batch

class _DataSetLoader(object):
    '''
    Internal class that represents an input queue with data from one of the DataSet objects.
    Each tower feeder will create and combine three data set loaders to one switchable queue.
    Keeps a ModelFeeder reference for accessing shared settings and placeholders.
    Keeps a DataSet reference to access its samples.
    '''
    def __init__(self, model_feeder, data_set, alphabet, dtype=tf.float32, logdir=''):
        self._model_feeder = model_feeder
        self._data_set = data_set
        max_queued_batches = 10
        self.queue = tf.PaddingFIFOQueue(shapes=[[None, None, model_feeder.numcep + (2 * model_feeder.numcep * model_feeder.numcontext)], [None,], [None,None,], [None,], [None,]],
                                         dtypes=[dtype, tf.int32, tf.int32, tf.int32, tf.int32],
                                         capacity=max_queued_batches)
        self._enqueue_op = self.queue.enqueue([model_feeder.ph_x, model_feeder.ph_x_length, model_feeder.ph_y, model_feeder.ph_flat_y, model_feeder.ph_y_length])
        self._close_op = self.queue.close(cancel_pending_enqueues=True)
        self._size_op = self.queue.size()
        self._size_summary = tf.summary.scalar('%s queue size' % data_set.name,
                                               tensor=self._size_op,
                                               collections=[])
        self._file_writer = tf.summary.FileWriter(os.path.join(logdir, data_set.name),
                                                  max_queue=100,
                                                  flush_secs=120)
        self._empty_op = self.queue.dequeue_many(self._size_op)
        self._alphabet = alphabet

    def start_queue_threads(self, session, coord):
        '''
        Starts concurrent queue threads for reading samples from the data set.
        '''
        queue_threads = [Thread(target=self._populate_batch_queue, args=(session, coord, self._data_set.name))
                         for i in range(self._model_feeder.threads_per_queue)]
        for queue_thread in queue_threads:
            coord.register_thread(queue_thread)
            queue_thread.daemon = True
            queue_thread.start()
        return queue_threads

    def empty_queue(self, session):
        session.run(self._empty_op)

    def close_queue(self, session):
        '''
        Closes the data set queue.
        '''
        session.run(self._close_op)
        self._file_writer.close()

    def _populate_batch_queue(self, session, coord, name):
        '''
        Queue thread routine.
        '''
        run_options = tf.RunOptions(timeout_in_ms=100)

        while not coord.should_stop():
            batch_x, batch_x_len = [], []
            batch_y, batch_y_len = [], []
            max_x, max_y = (0, 0)   # Used to pad each source before concat.

            indices = self._data_set.next_batch_indices()
            if indices is None:
                time.sleep(0.1)
                continue

            for index in indices:
                wav_file, transcript, _ = self._data_set.files[index]
                source = audiofile_to_input_vector(wav_file, self._model_feeder.numcep, self._model_feeder.numcontext)
                source_len = len(source)
                target = text_to_char_array(transcript, self._alphabet)
                target_len = len(target)
                if source_len < target_len:
                    raise ValueError('Error: Audio file {} is too short for transcription.'.format(wav_file))

                batch_x.append(source)
                batch_x_len.append(source_len)
                batch_y.append(target)
                batch_y_len.append(target_len)

                max_x = max(max_x, source_len)
                max_y = max(max_y, target_len)

            if not batch_x:
                return

            # Pad to max len and concat.
            padded_batch_x, padded_batch_y = [], []
            for x in batch_x:
                pad_x = max_x - len(x)
                if pad_x > 0:
                    x = np.pad(x, ((0, pad_x), (0, 0)), mode='constant')
                padded_batch_x.append(x)
            for y in batch_y:
                pad_y = max_y - len(y)
                if pad_y > 0:
                    y = np.pad(y, (0, pad_y), mode='constant')
                padded_batch_y.append(y)

            stack_x = np.stack(padded_batch_x)
            stack_y = np.stack(padded_batch_y)
            flat_y = np.concatenate(batch_y)

            queued = False
            while not queued:
                if coord.should_stop():
                    return
                try:
                    summary_str, _ = session.run([self._size_summary, self._enqueue_op],
                                                 feed_dict={ self._model_feeder.ph_x: stack_x,
                                                             self._model_feeder.ph_x_length: batch_x_len,
                                                             self._model_feeder.ph_y: stack_y,
                                                             self._model_feeder.ph_flat_y: flat_y,
                                                             self._model_feeder.ph_y_length: batch_y_len },
                                                 options=run_options)
                    self._file_writer.add_summary(summary_str)
                    queued = True
                except tf.errors.DeadlineExceededError:
                    continue
                except tf.errors.CancelledError:
                    return

class _TowerFeeder(object):
    '''
    Internal class that represents a switchable input queue for one tower.
    It creates, owns and combines three _DataSetLoader instances.
    Keeps a ModelFeeder reference for accessing shared settings and placeholders.
    '''
    def __init__(self, model_feeder, index, alphabet, dtype=tf.float32, logdir=''):
        self._model_feeder = model_feeder
        self.index = index
        self._loaders = [_DataSetLoader(model_feeder, data_set, alphabet, dtype, logdir) for data_set in model_feeder.sets]
        self._queues = [set_queue.queue for set_queue in self._loaders]
        self._queue = tf.QueueBase.from_list(model_feeder.ph_queue_selector, self._queues)
        self._close_op = self._queue.close(cancel_pending_enqueues=True)

    def next_batch(self):
        '''
        Draw the next batch from from the combined switchable queue.
        '''
        source, source_lengths, target, flat_target, target_lengths = self._queue.dequeue()
        sparse_labels = ctc_label_dense_to_sparse(target, target_lengths)
        return source, source_lengths, sparse_labels, flat_target, target_lengths

    def start_queue_threads(self, session, coord):
        '''
        Starts the queue threads of all owned _DataSetLoader instances.
        '''
        queue_threads = []
        for set_queue in self._loaders:
            queue_threads += set_queue.start_queue_threads(session, coord)
        return queue_threads

    def empty_queues(self, session):
        for set_queue in self._loaders:
            set_queue.empty_queue(session)

    def close_queues(self, session):
        '''
        Closes queues of all owned _DataSetLoader instances.
        '''
        for set_queue in self._loaders:
            set_queue.close_queue(session)
