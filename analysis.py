import numpy as np
import os
import miditoolkit
import pprint
import numpy as np
from miditoolkit.midi import parser as mid_parser  
from miditoolkit.midi import containers as ct
from operator import itemgetter
import argparse
from midi_processor.chord import detect_chord

parser = argparse.ArgumentParser()

parser.add_argument('--load_path', default="result/dec0722", help='모델 로드 경로', type=str)
parser.add_argument('--save_path', default='generated.mid', type=str)

args = parser.parse_args()

# set arguments
load_path = args.load_path
save_path= args.save_path

def analysis(path,num):

    acc_list = []

    for i in range(20):
        midi = miditoolkit.midi.parser.MidiFile(path+'/gen{}.mid'.format(num-20+i+1))
        #pprint.pprint(midi.instruments[0].notes)
        pitch_arr = np.zeros(12)

        for n in midi.instruments[0].notes:
            pitch_arr[n.pitch % 12] += 1

        acc = detect_key(pitch_arr)
        #print(i, acc)

        acc_list.append(acc)
        
    print(np.sum(np.array(acc_list))/20)
    return np.sum(np.array(acc_list))/20

def detect_key(arr):

    max = 0
    idx = 0
    arr_sum = np.sum(arr)

    for i in range(12):
        sum = arr[(i+0)%12] + arr[(i+2)%12] +arr[(i+4)%12] +arr[(i+5)%12] +arr[(i+7)%12] +arr[(i+9)%12] +arr[(i+11)%12]
        if sum > max:
            max = sum
            idx = i
    
    return max/arr_sum

acc = 0
acc += analysis(load_path,20)
acc += analysis(load_path,40)
acc += analysis(load_path,60)
acc += analysis(load_path,80)
acc += analysis(load_path,100)

#print(acc/5)