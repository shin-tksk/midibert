from model import MusicTransformerDecoder
from custom.layers import *
from custom import callback
import params as par
from tensorflow.python.keras.optimizer_v2.adam import Adam
from data import Data
import utils
import argparse
import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pickle

tf.executing_eagerly()

parser = argparse.ArgumentParser()

parser.add_argument('--l_r', default=None, type=float)
parser.add_argument('--batch_size', default=8, help='batch size', type=int)
parser.add_argument('--pickle_dirA', default='dataset/pop', help='데이터셋 경로')
parser.add_argument('--pickle_dirB', default='dataset/classic', help='데이터셋 경로')
parser.add_argument('--max_seq', default=2048, type=int)
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--load_path', default='model/230204', type=str)
parser.add_argument('--save_path', default="cla_model/230204")
parser.add_argument('--is_reuse', default=False)
parser.add_argument('--multi_gpu', default=True)
parser.add_argument('--num_layers', default=6, type=int)

args = parser.parse_args()

# set arguments
l_r = args.l_r
batch_size = args.batch_size
pickle_dirA = args.pickle_dirA
pickle_dirB = args.pickle_dirB
max_seq = args.max_seq
epochs = args.epochs
is_reuse = args.is_reuse
load_path = args.load_path
save_path = args.save_path
multi_gpu = args.multi_gpu
num_layer = args.num_layers

# load data
def batch_data(files,batch_size,length):
    batch_files = random.sample(files, k=batch_size)
    batch_data = []    
    label_data = []
    for file in batch_files:
        batch_data.append(padding(file, length))
        if file.split('/')[-2] == pickle_dirA.split('/')[-1]:
            label_data.append([1,0])
        elif file.split('/')[-2] == pickle_dirB.split('/')[-1]:
            label_data.append([0,1])
        else:
            print('error')
            
    return np.array(batch_data), np.array(label_data)

def padding(file,max_length):
    with open(file, 'rb') as f:
        data = pickle.load(f)
        if max_length <= len(data):
            start = random.randrange(0,len(data) - max_length)
            data = data[start:start + max_length]
        else:
            data = np.append(par.cls_token, data)
            while len(data) < max_length:
                data = np.append(data, par.pad_token)
    return data

datasetA = list(utils.find_files_by_extensions(pickle_dirA, ['.pickle']))
datasetB = list(utils.find_files_by_extensions(pickle_dirB, ['.pickle']))
data_len = min(len(datasetA), len(datasetB))
datasetA = datasetA[:data_len] 
datasetB = datasetB[:data_len]

train_dataA = datasetA[:int(len(datasetA) * 0.8)]
eval_dataA = datasetA[int(len(datasetA) * 0.8): int(len(datasetA) * 0.9)]
test_dataA = datasetA[int(len(datasetA) * 0.9):]

train_dataB = datasetB[:int(len(datasetB) * 0.8)]
eval_dataB = datasetB[int(len(datasetB) * 0.8): int(len(datasetB) * 0.9)]
test_dataB = datasetB[int(len(datasetB) * 0.9):]

train_data = train_dataA + train_dataB
eval_data = eval_dataA + eval_dataB
test_data = test_dataA + test_dataB

def train():
    mt = MusicTransformerDecoder(loader_path=load_path)
    _, _, look_ahead_mask = \
                    utils.get_masked_with_pad_tensor(decode_array.shape[1], decode_array, decode_array)
    result = mt.call(decode_array, lookup_mask=look_ahead_mask, training=False)