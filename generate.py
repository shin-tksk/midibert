from model import MusicTransformerDecoder
from custom.layers import *
from custom import callback
import params as par
#from tensorflow.python.keras.optimizer_v2.adam import Adam
from data import Data
import utils
import datetime
import argparse
import os
import pickle
from midi_processor.processor import decode_midi, encode_midi

tf.get_logger().setLevel("ERROR")

parser = argparse.ArgumentParser()

parser.add_argument('--max_seq', default=2048, help='최대 길이', type=int)
parser.add_argument('--load_path', default="model/230228/best", help='모델 로드 경로', type=str)
parser.add_argument('--length', default=2048, type=int)
parser.add_argument('--save_path', default='result/test/test1.mid', type=str)
parser.add_argument('--prior_midi', default='midi/yoru/shoes.mid', type=str, help='prior data for generate midi file')

args = parser.parse_args()

# set arguments
max_seq = args.max_seq
load_path = args.load_path
length = args.length
save_path= args.save_path
prior_midi = args.prior_midi

mt = MusicTransformerDecoder(loader_path=load_path)

fname = 'dataset/2048/yoru/shoes.mid_3.pickle'
with open(fname, 'rb') as f:
    data = pickle.load(f)

inputs = [1,3] + [2] * 2046
#decode_midi(data, 'result/pitchall/ori.mid')
for i in range(1):
    result = mt.generate_mask_pro(inputs, length=len(inputs))
    decode_midi(result, 'result/mask_pro/gen{}.mid'.format(i))
    #np.save('result/mask_pro/gen{}.npy'.format(i), np.array(result))
#print(result)
exit()
inputs = [1] + data
idx_list = []
for i,w in enumerate(inputs):
    if 3 < w < 52:
        idx_list.append(i)
print(len(idx_list))
result = mt.generate(inputs, length=len(inputs), idx_list=idx_list)
decode_midi(result, 'result/pitchall/gen.mid')
exit()

#inputs = [1] + encode_midi(prior_midi) 
#inputs = [1,3]
decode_midi(inputs, 'result/mask/mid/origin.mid')
for i in range(5):
    result = mt.generate_mask(inputs, length=len(inputs))
    decode_midi(result, 'result/mask/mid/gen{}.mid'.format(i))
    np.save('result/mask/npy/gen{}.npy'.format(i), np.array(result))
    inputs = result
    
