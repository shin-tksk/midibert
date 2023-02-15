import pickle
import os
import re
import sys
import hashlib
from progress.bar import Bar
import tensorflow as tf
import utils
import params as par
from midi_processor.processor import encode_midi, decode_midi
from midi_processor import processor
import random

#0679868b152984be7b3b73a1ca5b204e.mid

if __name__ == '__main__':
    words = encode_midi('midi/yoru/just.mid')
    #print(words[:10])
    decode_midi(words,'result/test/gen.mid')
