from midi_processor.processor import encode_midi, decode_midi
import numpy as np
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

#if __name__ == '__main__':
    #words = encode_midi('midi/lmd_full/1/10e903c3aa7a6b6c5ce5a74d5cfb8702.mid')
    #print(len(words))
    #print(words[:10])
    #decode_midi(words,'result/test/gen.mid')

#words = encode_midi('midi/lmd_full/2/29ef5363a4fdd625ae1aed25b881b21e.mid')
#print(words)
#exit()
root = 'midi/yoru'
l = [f for f in os.listdir(root)]

def divide(words,max,p):
    i = max-1
    while words[i] != 3:
        i -= 1
    if i != 0:
        result = words[:i]
        words = words[i:]
    else:
        result = words[:max]
        words = words[max:]
        if 3 in words:
            idx = words.index(3)
            words = words[idx:]
        else:
            return [0]

    if 0 < len(result) <= max:
        if result.count(3) >= 50:
            print(len(result),result.count(3),result[-1],p)
            #print(result.count(3))
        else:
            print(result[:20])
    return words

for p in l:
    words = encode_midi(root+'/'+p)
    if words == None:
        continue
    while(len(words) > 1023):
        words = divide(words, 1023, p)