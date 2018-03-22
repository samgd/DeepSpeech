import numpy as np
import pandas
import random
import tensorflow as tf

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
                 threads_per_queue=2,
                 dtype=tf.float32):

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
        self.ph_y_length = tf.placeholder(tf.int32, [None,])
        self.ph_batch_size = tf.placeholder(tf.int32, [])
        self.ph_queue_selector = tf.placeholder(tf.int32, name='Queue_Selector')

        self._tower_feeders = [_TowerFeeder(self, i, alphabet, dtype) for i in range(self.tower_feeder_count)]

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
        feed_dict[self.ph_batch_size] = data_set.batch_size

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
    def __init__(self, name, csvs, batch_size, skip=0, limit=0, ascending=True, next_index=lambda i: i + 1, shuffle_batch_order=False, shuffle_seed=1234):
        self.name = name
        self.batch_size = batch_size
        self.next_index = next_index
        self.files = None
        for csv in csvs:
            file = pandas.read_csv(csv, encoding='utf-8')
            if self.files is None:
                self.files = file
            else:
                self.files = self.files.append(file)
        self.files = self.files.sort_values(by="wav_filesize", ascending=ascending) \
                         .ix[:, ["wav_filename", "transcript"]] \
                         .values[skip:]
        if limit > 0:
            self.files = self.files[:limit]
        self.total_batches = int(ceil(float(len(self.files)) / batch_size))

        all_indices = list(range(len(self.files)))
        self.batch_indices = [all_indices[i*batch_size:(i + 1)*batch_size]
                              for i in range(self.total_batches)]
        self.current_batch = -1
        self.n_batch = 0
        self.shuffle_batch_order = shuffle_batch_order
        self.shuffle_seed = shuffle_seed
        self._lock = Lock()

    def next_batch_indices(self):
        with self._lock:
            idx = self.next_index(self.current_batch)

            if idx >= self.total_batches:
                if self.n_batch % self.total_batches == 0 and self.shuffle_batch_order:
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
    def __init__(self, model_feeder, data_set, alphabet, dtype=tf.float32):
        self._model_feeder = model_feeder
        self._data_set = data_set
        max_queued_batches = 30
        self.queue = tf.PaddingFIFOQueue(shapes=[[None, None, model_feeder.numcep + (2 * model_feeder.numcep * model_feeder.numcontext)], [None,], [None,None,], [None,]],
                                         dtypes=[dtype, tf.int32, tf.int32, tf.int32],
                                         capacity=max_queued_batches)
        self._enqueue_op = self.queue.enqueue([model_feeder.ph_x, model_feeder.ph_x_length, model_feeder.ph_y, model_feeder.ph_y_length])
        self._close_op = self.queue.close(cancel_pending_enqueues=True)
        self._size_op = self.queue.size()
        tf.summary.scalar('%s queue size' % data_set.name, self._size_op)
        self._empty_op = self.queue.dequeue_many(self._size_op)
        self._alphabet = alphabet

    def start_queue_threads(self, session, coord):
        '''
        Starts concurrent queue threads for reading samples from the data set.
        '''
        queue_threads = [Thread(target=self._populate_batch_queue, args=(session, coord))
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

    def _populate_batch_queue(self, session, coord):
        '''
        Queue thread routine.
        '''
        run_options = tf.RunOptions(timeout_in_ms=10000)

        while not coord.should_stop():
            batch_x, batch_x_len = [], []
            batch_y, batch_y_len = [], []
            max_x, max_y = (0, 0)   # Used to pad each source before concat.

            for index in self._data_set.next_batch_indices():
                wav_file, transcript = self._data_set.files[index]
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

            queued = False
            while not queued and not coord.should_stop():
                try:
                    session.run(self._enqueue_op,
                                feed_dict={ self._model_feeder.ph_x: np.stack(padded_batch_x),
                                            self._model_feeder.ph_x_length: batch_x_len,
                                            self._model_feeder.ph_y: np.stack(padded_batch_y),
                                            self._model_feeder.ph_y_length: batch_y_len },
                                options=run_options)
                    queued = True
                except tf.errors.DeadlineExceededError:
                    pass
                except tf.errors.CancelledError:
                    return

class _TowerFeeder(object):
    '''
    Internal class that represents a switchable input queue for one tower.
    It creates, owns and combines three _DataSetLoader instances.
    Keeps a ModelFeeder reference for accessing shared settings and placeholders.
    '''
    def __init__(self, model_feeder, index, alphabet, dtype=tf.float32):
        self._model_feeder = model_feeder
        self.index = index
        self._loaders = [_DataSetLoader(model_feeder, data_set, alphabet, dtype) for data_set in model_feeder.sets]
        self._queues = [set_queue.queue for set_queue in self._loaders]
        self._queue = tf.QueueBase.from_list(model_feeder.ph_queue_selector, self._queues)
        self._close_op = self._queue.close(cancel_pending_enqueues=True)

    def next_batch(self):
        '''
        Draw the next batch from from the combined switchable queue.
        '''
        source, source_lengths, target, target_lengths = self._queue.dequeue()
        sparse_labels = ctc_label_dense_to_sparse(target, target_lengths)
        return source, source_lengths, sparse_labels

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
