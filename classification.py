from model import MusicTransformerDecoder
from custom.layers import *
from custom import callback
import params as par
#from tensorflow.python.keras.optimizer_v2.adam import Adam
from data import Data
import utils
import datetime
import argparse
import pickle
import os

tf.get_logger().setLevel("ERROR")
parser = argparse.ArgumentParser()

parser.add_argument('--pickle_dirA', default='dataset/classic', help='데이터셋 경로')
parser.add_argument('--pickle_dirB', default='dataset/pop', help='데이터셋 경로')
parser.add_argument('--max_seq', default=2048, type=int)
parser.add_argument('--load_path', default='finetune/230228/CandP', type=str)

args = parser.parse_args()

# set arguments

pickle_dirA = args.pickle_dirA
pickle_dirB = args.pickle_dirB
max_seq = args.max_seq
load_path = args.load_path

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
test = testA[:min(len(testA),len(testB))] + testB[:min(len(testA),len(testB))]
#eval = evalA[:min(len(evalA),len(evalB))] + evalB[:min(len(evalA),len(evalB))]
#mt = MusicTransformerDecoder(loader_path=load_path)
print(pickle_dirA.split('/')[1],'and',pickle_dirB.split('/')[1])
print(len(test))
#count = 0
#mt = MusicTransformerDecoder(loader_path=load_path+'/0/best')
for f in range(3):
    count = 0
    mt = MusicTransformerDecoder(loader_path=load_path+'/{}/best'.format(f))
    for i,d in enumerate(test):
        data = padding(d['data'])
        data = tf.constant([data])
        if d['label'] == pickle_dirA.split('/')[1]:
            label = np.array([1,0])
        elif d['label'] == pickle_dirB.split('/')[1]:
            label = np.array([0,1])
        result = np.zeros(2)
        result += mt.classification(data)
        #print(result,label)
        if (result[0] < result[1]) == (label[0] < label[1]):
            count += 1
        #print(i,count)
        #break
    print('accuracy', count/len(test))