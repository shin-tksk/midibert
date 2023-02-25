import utils
import random
import pickle
#from tensorflow.python import keras
import numpy as np
import params as par
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]= "true"

class Data:
    def __init__(self, dir_path):
        self.files = list(utils.find_files_by_extensions(dir_path, ['.pickle']))
        #print(len(self.files))
        self.file_dict = {
            'train': self.files[:int(len(self.files) * 0.8)],
            'eval': self.files[int(len(self.files) * 0.8): int(len(self.files) * 0.9)],
            'test': self.files[int(len(self.files) * 0.9):],
        }
        self._seq_file_name_idx = 0
        self._seq_idx = 0
        pass

    def __repr__(self):
        return '<class Data has "'+str(len(self.files))+'" files>'

    def batch(self, batch_size, length, mode='train'):
        batch_files = random.sample(self.file_dict[mode], k=batch_size)
        #print(batch_files)
        batch_data = [
            self._get_seq(file, length, mask=False)
            for file in batch_files
        ]

        mask_data = [
            self._get_seq(file, length, mask=True)
            for file in batch_files
        ]

        return np.array(mask_data), np.array(batch_data)  # batch_size, seq_len

    def slide_seq2seq_batch(self, batch_size, length, mode='train'):
        x,y = self.batch(batch_size, length, mode)
        return x, y

    def _get_seq(self, fname, max_length=None, mask=False):
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        #print(data[:20])
        if max_length is not None:
            if max_length <= len(data):
                start = random.randrange(0,len(data) - max_length)
                data = data[start:start + max_length]
            else:
                data = np.append(par.cls_token, data)
                while len(data) < max_length:
                    data = np.append(data, par.pad_token)
                if mask:
                    data = utils.data_mask(data)
        return data

if __name__ == '__main__':
    import pprint
    def count_dict(max_length, data):
        cnt_arr = [0] * max_length
        cnt_dict = {}
        # print(cnt_arr)
        for batch in data:
            for index in batch:
                try:
                    cnt_arr[int(index)] += 1

                except:
                    print(index)
                try:
                    cnt_dict['index-'+str(index)] += 1
                except KeyError:
                    cnt_dict['index-'+str(index)] = 1
        return cnt_arr

    # print(add_noise(np.array([[1,2,3,3,4,5,6]]), rate=0.2))


    # print(par.vocab_size)
    # data = Data('dataset/processed')
    # # ds = DataSequence('dataset/processed', 10, 2048)
    # sample = data.seq2seq_batch(1000, 100)[0]
    # pprint.pprint(list(sample))
    # arr = count_dict(par.vocab_size+3,sample)
    # pprint.pprint(
    #     arr)
    #
    # from sequence import EventSeq, Event
    #
    # event_cnt = {
    #     'note_on': 0,
    #     'note_off': 0,
    #     'velocity': 0,
    #     'time_shift': 0
    # }
    # for event_index in range(len(arr)):
    #     for event_type, feat_range in EventSeq.feat_ranges().items():
    #
    #         if feat_range.start <= event_index < feat_range.stop:
    #             print(event_type+':'+str(arr[event_index])+' event cnt: '+str(event_cnt))
    #             event_cnt[event_type] += arr[event_index]
    #
    # print(event_cnt)

    # print(np.max(sample), np.min(sample))
    # print([data._get_seq(file).shape for file in data.files])
    #while True:
    # print(ds.__getitem__(10)[1].argmax(-1))