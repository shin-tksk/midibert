import numpy as np
import os
import pprint
import numpy as np
from operator import itemgetter
import math
from scipy.spatial import distance

transform = []

 # transform matrix
for n in range(12):
    transform.append([math.sin(math.radians(n*210)),math.cos(math.radians(n*210)),
        math.sin(math.radians(n*270)),math.cos(math.radians(n*270)),
        math.sin(math.radians(n*120)),math.cos(math.radians(n*120)),])
transform = np.array(transform).reshape(12,6) 
     
major = np.load('midi_processor/chord_npy/major.npy')
minor = np.load('midi_processor/chord_npy/minor.npy')
mb5 = np.load('midi_processor/chord_npy/mb5.npy')
aug = np.load('midi_processor/chord_npy/aug.npy')
seventh = np.load('midi_processor/chord_npy/seventh.npy')
minor7 = np.load('midi_processor/chord_npy/minor7.npy')
major7 = np.load('midi_processor/chord_npy/major7.npy')
m7b5 = np.load('midi_processor/chord_npy/m7b5.npy')
dim7 = np.load('midi_processor/chord_npy/dim7.npy')
add9 = np.load('midi_processor/chord_npy/add9.npy')
madd9 = np.load('midi_processor/chord_npy/madd9.npy')

root_name = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
    
def detect_chord(chord):

    sum = np.sum(chord)

    if sum == 0:
        return None

    else:
        chroma = chord / sum
        tonnetz = np.dot(chroma,transform)
        threshold = 10000

        for i, c in enumerate(major):
            dis = distance.euclidean(c,tonnetz)
            if dis < threshold:
                threshold = dis
                #name = root_name[i]
                chord_num = i

        for i, c in enumerate(minor):
            dis = distance.euclidean(c,tonnetz)
            if dis < threshold:
                threshold = dis
                #name = root_name[i]+'m'
                chord_num = i + 12
        '''
        for i, c in enumerate(mb5):
            dis = distance.euclidean(c,tonnetz)
            if dis < threshold:
                threshold = dis
                #name = root_name[i]+'mb5'
                chord_num = None

        for i, c in enumerate(aug):
            dis = distance.euclidean(c,tonnetz)
            if dis < threshold:
                threshold = dis
                #name = root_name[i]+'aug'
                chord_num = None

        for i, c in enumerate(seventh):
            dis = distance.euclidean(c,tonnetz)
            if dis < threshold:
                threshold = dis
                #name = root_name[i]+'7'
                chord_num = i

        for i, c in enumerate(minor7):
            dis = distance.euclidean(c,tonnetz)
            if dis < threshold:
                threshold = dis
                #name = root_name[i]+'m7'
                chord_num = i + 12

        for i, c in enumerate(major7):
            dis = distance.euclidean(c,tonnetz)
            if dis < threshold:
                threshold = dis
                #name = root_name[i]+'M7'
                chord_num = i

        for i, c in enumerate(m7b5):
            dis = distance.euclidean(c,tonnetz)
            if dis < threshold:
                threshold = dis
                #name = root_name[i]+'m7b5'
                chord_num = None

        for i, c in enumerate(dim7):
            dis = distance.euclidean(c,tonnetz)
            if dis < threshold:
                threshold = dis
                #name = root_name[i]+'dim7'
                chord_num = None

        for i, c in enumerate(add9):
            dis = distance.euclidean(c,tonnetz)
            if dis < threshold:
                threshold = dis
                #name = root_name[i]+'add9'
                chord_num = i

        for i, c in enumerate(madd9):
            dis = distance.euclidean(c,tonnetz)
            if dis < threshold:
                threshold = dis
                #name = root_name[i]+'madd9'
                chord_num = i + 12
        '''
        return chord_num

