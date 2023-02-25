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
from midi_processor.processor import decode_midi, encode_midi

tf.get_logger().setLevel("ERROR")

parser = argparse.ArgumentParser()

parser.add_argument('--max_seq', default=1024, help='최대 길이', type=int)
parser.add_argument('--load_path', default="model/230220", help='모델 로드 경로', type=str)
parser.add_argument('--length', default=1024, type=int)
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

#inputs = [1] + encode_midi(prior_midi) 
inputs = [1,3]

#result = mt.generate(inputs[:512], length=512)
#decode_midi(result, 'result/test/test.mid')
for i in range(3):
    result = mt.generate(inputs[:2], length=512)
    print(result[:20])
    decode_midi(result, 'result/test/mid/gen{}.mid'.format(i))
    np.save('result/test/npy/gen{}.npy'.format(i), np.array(result))
    #inputs = result
