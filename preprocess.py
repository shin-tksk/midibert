import pickle
import os
import re
import sys
from progress.bar import Bar
#import tensorflow as tf
from utils import find_files_by_extensions
import params as par
from midi_processor.processor import encode_midi, decode_midi
from midi_processor import processor
import random

def idx_search(words, max, count, save_dir, path):

    i = max-1
    while words[i] != 3:
        i -= 1

    if i != 0:
        result = words[:i]
        words = words[i:]
    else:
        result = words[:max]
        words = words[max:]
        if 3 in words:
            idx = words.index(3)
            words = words[idx:]
        else:
            return [0]

    if 0 < len(result) <= max and result.count(3) <= 50:
        with open('{}/{}_{}.pickle'.format(save_dir, path.split('/')[-1], count), 'wb') as f:
                    pickle.dump(result, f)

    else:
        print()
        print(path)
    
    return words
    
def divide(words, max, save_dir, path):
    #print('ok')
    count = 0
    if words == None:
        return
    while(len(words) > max):
        #print(count)
        words = idx_search(words, max, count, save_dir, path)
        count += 1
    result = words
    return

def preprocess_midi(path):
    return encode_midi(path)

def preprocess_midi_files_under(midi_root, save_dir, max):
    midi_paths = list(find_files_by_extensions(midi_root, ['.mid', '.midi']))
    os.makedirs(save_dir, exist_ok=True)

    for path in Bar('Processing').iter(midi_paths):
        print(' ', end='[{}]'.format(path), flush=True)

        data = preprocess_midi(path)
        try:
            divide(data, int(max), save_dir, path)
        except:
            continue
        #break
    return


class TFRecordsConverter(object):
    def __init__(self, midi_path, output_dir,
                 num_shards_train=3, num_shards_test=1):
        self.output_dir = output_dir
        self.num_shards_train = num_shards_train
        self.num_shards_test = num_shards_test

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Get lists of ev_seq and ctrl_seq
        self.es_seq_list, self.ctrl_seq_list = self.process_midi_from_dir(midi_path)

        # Counter for total number of images processed.
        self.counter = 0
    pass

    def process_midi_from_dir(self, midi_root):

        midi_paths = list(utils.find_files_by_extensions(midi_root, ['.mid', '.midi', '.MID']))
        es_seq_list = []
        ctrl_seq_list = []
        for path in Bar('Processing').iter(midi_paths):
            print(' ', end='[{}]'.format(path), flush=True)

            try:
                data = preprocess_midi(path)
                for es_seq, ctrl_seq in data:
                    max_len = par.max_seq
                    for idx in range(max_len + 1):
                        es_seq_list.append(data[0])
                        ctrl_seq_list.append(data[1])

            except KeyboardInterrupt:
                print(' Abort')
                return
            except:
                print(' Error')
                continue

        return es_seq_list, ctrl_seq_list

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def __write_to_records(self, output_path, indicies):
        writer = tf.io.TFRecordWriter(output_path)
        for i in indicies:
            es_seq = self.es_seq_list[i]
            ctrl_seq = self.ctrl_seq_list[i]

        # example = tf.train.Example(features=tf.train.Features(feature={
        #         'label': TFRecordsConverter._int64_feature(label),
        #         'text': TFRecordsConverter._bytes_feature(bytes(x, encoding='utf-8'))}))


if __name__ == '__main__':
    preprocess_midi_files_under(
            midi_root=sys.argv[1],
            save_dir=sys.argv[2],
            max=sys.argv[3])
