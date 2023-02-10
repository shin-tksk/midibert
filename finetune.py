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
import pickle
import random

tf.executing_eagerly()

parser = argparse.ArgumentParser()

parser.add_argument('--l_r', default=None, type=float)
parser.add_argument('--batch_size', default=8, help='batch size', type=int)
parser.add_argument('--pickle_dirA', default='dataset/pop', help='데이터셋 경로')
parser.add_argument('--pickle_dirB', default='dataset/classic', help='데이터셋 경로')
parser.add_argument('--max_seq', default=2048, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--save_path', default="finetune")
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

if load_path == None:
    print('Load path not specified')

# load data
def load_data(pickle_dir):
    train = list(utils.find_files_by_extensions(pickle_dir+'/train', ['.pickle']))
    test = list(utils.find_files_by_extensions(pickle_dir+'/test', ['.pickle']))
    eval = list(utils.find_files_by_extensions(pickle_dir+'/eval', ['.pickle']))
    return load_file(train), load_file(test), load_file(eval)

def load_file(files):
    dataset = [] 
    for fname in files:
        with open(fname, 'rb') as f:
            data = pickle.load(f)
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
    for data in batch_files:
        batch_data.append(padding(data['data']))
        if data['label'] == pickle_dirA.split('/')[1]:
            label_data.append([1,0])
        elif data['label'] == pickle_dirB.split('/')[1]:
            label_data.append([0,1])
        else:
            print('error')
    return np.array(batch_data), np.array(label_data)

trainA, testA, evalA = load_data(pickle_dirA)
trainB, testB, evalB = load_data(pickle_dirB)
train = trainA[:min(len(trainA),len(trainB))] + trainB[:min(len(trainA),len(trainB))]
batch_x, batch_y = make_batch(train)
#print(batch_x)

# load model
learning_rate = callback.CustomSchedule(par.embedding_dim) if l_r is None else l_r
opt = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# define model
mt = MusicTransformerDecoder(
            embedding_dim=256,
            vocab_size=par.vocab_size,
            num_layer=num_layer,
            max_seq=max_seq,
            dropout=0.2,
            debug=False, loader_path=load_path)
mt.compile(optimizer=opt, loss=callback.classification_loss)
print(mt)
exit()
# define tensorboard writer
current_time = datetime.datetime.now().strftime('%Y%m%d')
train_log_dir = 'logs/finetune/'+current_time+'/train'
eval_log_dir = 'logs/finetune/'+current_time+'/eval'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)

# Train Start
idx = 0
best = 0
print('------------------train start--------------------')

for e in range(epochs):

    mt.reset_metrics()

    for b in range(len(dataset.files) // batch_size):
        try:
            batch_x, batch_y = dataset.slide_seq2seq_batch(batch_size, max_seq)
        except:
            continue

        result_metrics = mt.train_on_batch(batch_x, batch_y)
        eval_x, eval_y = dataset.slide_seq2seq_batch(batch_size, max_seq, 'eval')
        eval_result_metrics, weights = mt.evaluate(eval_x, eval_y)
        
        
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
        
        if eval_result_metrics[1] > best:
            best = eval_result_metrics[1]
            mt.save_weight(save_path)
        #break

    if (e + 1) % 10 == 0:
        os.makedirs(save_path + '/{}epoch'.format(e + 1), exist_ok=True)
        mt.save_weight(save_path)

print('best :',best)

