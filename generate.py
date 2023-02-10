from model import MusicTransformerDecoder
from custom.layers import *
from custom import callback
import params as par
from tensorflow.python.keras.optimizer_v2.adam import Adam
from data import Data
import utils
import datetime
import argparse
import os
from midi_processor.processor import decode_midi, encode_midi


parser = argparse.ArgumentParser()

parser.add_argument('--max_seq', default=2048, help='최대 길이', type=int)
parser.add_argument('--load_path', default="model/230118", help='모델 로드 경로', type=str)
parser.add_argument('--mode', default='dec')
parser.add_argument('--beam', default=None, type=int)
parser.add_argument('--length', default=128, type=int)
parser.add_argument('--save_path', default='generated.mid', type=str)
parser.add_argument('--prior_midi', default='midi/yoru/just.mid', type=str, help='prior data for generate midi file')
parser.add_argument('--famous', default=True, type=bool)

args = parser.parse_args()


# set arguments
max_seq = args.max_seq
load_path = args.load_path
mode = args.mode
beam = args.beam
length = args.length
save_path= args.save_path
famous = args.famous

prior_midi = args.prior_midi
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
gen_log_dir = 'logs/mt_decoder/generate_'+current_time+'/generate'
gen_summary_writer = tf.summary.create_file_writer(gen_log_dir)


if mode == 'enc-dec':
    print(">> generate with original seq2seq wise... beam size is {}".format(beam))
    mt = MusicTransformer(
            embedding_dim=256,
            vocab_size=par.vocab_size,
            num_layer=6,
            max_seq=2048,
            dropout=0.2,
            debug=False, loader_path=load_path)
else:
    print(">> generate with decoder wise... beam size is {}".format(beam))
    mt = MusicTransformerDecoder(loader_path=load_path)

inputs = encode_midi(prior_midi) 
    
if famous:
    os.makedirs('result/famous', exist_ok=True)

    bar_list = []

    for i, w in enumerate(inputs):
        if w == 0:
            bar_list.append(i)

    start = 13
    end = 17
    select = inputs[bar_list[start-2]+1:bar_list[end-1]+1]
    select.insert(0,inputs[0])

    with gen_summary_writer.as_default():
        result = mt.generate(inputs[:1], beam=beam, length=length, tf_board=True)
        #result = mt.generate(select, beam=beam, length=length, tf_board=True)

    if mode == 'enc-dec':
        decode_midi(list(inputs[-1*par.max_seq:]) + list(result[1:]), file_path=save_path)
    else:
        decode_midi(result, 'result/famous/'+prior_midi.split('/')[-1])

else:
    os.makedirs('result/'+save_path, exist_ok=True)

    for i in range(5):

        with gen_summary_writer.as_default():
            result = mt.generate(inputs[:1], beam=beam, length=length, tf_board=True)

        #result = np.delete(result,0)

        if mode == 'enc-dec':
            decode_midi(list(inputs[-1*par.max_seq:]) + list(result[1:]), file_path=save_path)
        else:
            decode_midi(result, 'result/'+ save_path +'/gen{}.mid'.format(i+1))