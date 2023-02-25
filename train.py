from model import MusicTransformerDecoder
from custom.layers import *
from custom import callback
import params as par
#from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.optimizers import RectifiedAdam as RAdam
from data import Data
import utils
import argparse
import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import time
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]= "true"

tf.executing_eagerly()

parser = argparse.ArgumentParser()

parser.add_argument('--l_r', default=None, type=float)
parser.add_argument('--batch_size', default=32, help='batch size', type=int)
parser.add_argument('--pickle_dir', default='dataset/lmd/0_2', help='데이터셋 경로')
parser.add_argument('--max_seq', default=1024, type=int)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--save_path', default="model/230224")
parser.add_argument('--num_layers', default=8, type=int)

args = parser.parse_args()

# set arguments
l_r = args.l_r
batch_size = args.batch_size
pickle_dir = args.pickle_dir
max_seq = args.max_seq
epochs = args.epochs
load_path = args.load_path
save_path = args.save_path
num_layer = args.num_layers

# load data
dataset = Data(pickle_dir)
print('data num :',len(dataset.files))

# load model
learning_rate = callback.CustomSchedule(par.embedding_dim) if l_r is None else l_r
opt = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
#opt = RAdam(learning_rate=0.001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
#opt = RAdam()

# define model
mt = MusicTransformerDecoder(
            embedding_dim=128,
            vocab_size=par.vocab_size,
            num_layer=num_layer,
            max_seq=max_seq,
            dropout=0.2,
            loader_path=load_path)
mt.compile(optimizer=opt, loss=callback.transformer_dist_train_loss)


# define tensorboard writer
current_time = datetime.datetime.now().strftime('%Y%m%d')
train_log_dir = 'logs/mt_decoder/'+current_time+'/train'
eval_log_dir = 'logs/mt_decoder/'+current_time+'/eval'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)

# Train Start
idx = 0
best = 10

print('------------------train start--------------------')
s = time.time()
for e in range(epochs):

    mt.reset_metrics()

    for b in range(len(dataset.files) // batch_size):
        try:
            batch_x, batch_y = dataset.slide_seq2seq_batch(batch_size, max_seq)
        except:
            continue

        bs = time.time()
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
        print('time :',time.time()-bs)

        if eval_result_metrics[0] < best:
            os.makedirs(save_path + '/best', exist_ok=True)
            best = eval_result_metrics[0]
            mt.save_weight(save_path + '/best')

    if (e + 1) % 10 == 0:
        os.makedirs(save_path + '/{}epoch'.format(e + 1), exist_ok=True)
        mt.save_weight(save_path)

    mt.save_weight(save_path)
en = time.time()
print('best :',best)
print('time :',en-s)


