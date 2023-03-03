from model import MusicTransformerDecoder
from custom.layers import *
from custom import callback
import params as par
#from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.keras.optimizers import Adam
from data import Data
import utils
import argparse
import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import random
import time

tf.executing_eagerly()

parser = argparse.ArgumentParser()

parser.add_argument('--l_r', default=0.0001, type=float)
parser.add_argument('--batch_size', default=16, help='batch size', type=int)
parser.add_argument('--pickle_dirA', default='dataset/classic', help='데이터셋 경로')
parser.add_argument('--pickle_dirB', default='dataset/pop', help='데이터셋 경로')
parser.add_argument('--max_seq', default=2048, type=int)
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--load_path', default='model/230228/best', type=str)
parser.add_argument('--save_path', default="finetune/230228/CandP")
parser.add_argument('--num_layers', default=4, type=int)

args = parser.parse_args()

# set arguments
l_r = args.l_r
batch_size = args.batch_size
pickle_dirA = args.pickle_dirA
pickle_dirB = args.pickle_dirB
max_seq = args.max_seq
epochs = args.epochs
load_path = args.load_path
save_path = args.save_path
num_layer = args.num_layers

if load_path == None:
    print('Load path not specified')

# load data
def load_data(pickle_dir):
    train = list(utils.find_files_by_extensions(pickle_dir+'/train', ['.pickle']))
    eval = list(utils.find_files_by_extensions(pickle_dir+'/eval', ['.pickle']))
    return load_file(train), load_file(eval)

def load_file(files):
    dataset = [] 
    for fname in files:
        print(fname)
        with open(fname, 'rb') as f:
            data = pickle.load(f)
            if len(data) < max_seq and data[0]==3:
                dataset.append({'data':data, 'label':fname.split('/')[1]})
    return dataset

def padding(data):
    data = np.append(par.cls_token,data)
    while len(data) < max_seq:
        data = np.append(data, par.pad_token)
    return data

def make_batch(data):
    batch_files = random.sample(data, k=batch_size)
    batch_data = []
    label_data = []
    real_data = []
    for data in batch_files:
        real_data.append(padding(data['data']))
        batch_data.append(utils.data_mask(padding(data['data'])))
        if data['label'] == pickle_dirA.split('/')[1]:
            label_data.append([1,0])
        elif data['label'] == pickle_dirB.split('/')[1]:
            label_data.append([0,1])
        else:
            print('error')
    return np.array(batch_data), np.array(label_data), np.array(real_data)

trainA, evalA = load_data(pickle_dirA)
trainB, evalB = load_data(pickle_dirB)
train = trainA[:min(len(trainA),len(trainB))] + trainB[:min(len(trainA),len(trainB))]
eval = evalA[:min(len(evalA),len(evalB))] + evalB[:min(len(evalA),len(evalB))]
#batch_x, batch_y, real = make_batch(train)
#print(real[:,:10])
#exit()
# load model
learning_rate = callback.CustomSchedule(par.embedding_dim) if l_r is None else l_r
opt = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# define model
mt = MusicTransformerDecoder(
            embedding_dim=128,
            vocab_size=par.vocab_size,
            num_layer=num_layer,
            max_seq=max_seq,
            dropout=0.2,
            loader_path=load_path)
mt.compile(optimizer=opt, loss=callback.classification_loss)

# define tensorboard writer
current_time = datetime.datetime.now().strftime('%Y%m%d')
train_log_dir = 'logs/finetune/'+current_time+'/train'
eval_log_dir = 'logs/finetune/'+current_time+'/eval'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)

# Train Start

idx = 0
best = 10
print('------------------train start--------------------')
s = time.time()

for f in range(3):
    idx = 0
    best = 10
    os.makedirs(save_path + '/{}'.format(f), exist_ok=True)
    for e in range(epochs):

        mt.reset_metrics()

        for b in range(len(train) // batch_size):
            try:
                batch_x, batch_y, real = make_batch(train)
                #print(batch_x)
            except:
                continue
            #utils.bar_id(batch_x)
            result_metrics = mt.cla_train_on_batch(batch_x, batch_y, real)
            eval_x, eval_y, real = make_batch(eval)
            eval_result_metrics = mt.cla_evaluate(eval_x, eval_y, real)
            
            
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', result_metrics[0], step=idx)
                tf.summary.scalar('accuracy', result_metrics[1], step=idx)

            with eval_summary_writer.as_default():
                tf.summary.scalar('loss', eval_result_metrics[0], step=idx)
                tf.summary.scalar('accuracy', eval_result_metrics[1], step=idx)
            
            idx += 1

            print('\n====================================================')
            print('Epoch/Batch: {}/{}'.format(e, b))
            print('Train >>>> Loss: {:6.6}, Accuracy: {}'.format(result_metrics[0], result_metrics[1]))
            print('Eval >>>> Loss: {:6.6}, Accuracy: {}'.format(eval_result_metrics[0], eval_result_metrics[1]))
            
            if eval_result_metrics[0] + result_metrics[0] < best:
                os.makedirs(save_path + '/{}/best'.format(f), exist_ok=True)
                best = eval_result_metrics[0] + result_metrics[0]
                mt.save_weight(save_path + '/{}/best'.format(f))

    mt.save_weight(save_path + '/{}'.format(f))
#print('best :',best)
en = time.time()
print('best :',best)
print('time :',en-s)
