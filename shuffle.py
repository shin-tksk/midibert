from random import shuffle
import numpy as np
import os
import shutil

test_ratio = 0.1
ROOT_PATH = ''

def shuffle_data(genru):
    #os.rename('dataset/{}'.format(genru), 'dataset/{}_train'.format(genru,genru))
    os.makedirs(os.path.join(ROOT_PATH, 'dataset/{}/test'.format(genru)), exist_ok=True)
    os.makedirs(os.path.join(ROOT_PATH, 'dataset/{}/eval'.format(genru)), exist_ok=True)
    
    #shutil.copytree(os.path.join(ROOT_PATH, 'dataset/{}_data'.format(genru)), os.path.join(ROOT_PATH, 'dataset/{}/{}_train'.format(genru,genru)))

    midi = [f for f in os.listdir(os.path.join(ROOT_PATH, 'dataset/{}/train'.format(genru)))]
    length = len(midi)
    print(length)
    idx = np.random.choice(len(midi), int(test_ratio * length), replace=False)
    print(len(idx))
    for i in idx:
        shutil.move(os.path.join(ROOT_PATH, 'dataset/{}/train/{}'.format(genru,midi[i])),
                os.path.join(ROOT_PATH, 'dataset/{}/test/{}'.format(genru,midi[i])))
    
    midi = [f for f in os.listdir(os.path.join(ROOT_PATH, 'dataset/{}/train'.format(genru)))]
    idx = np.random.choice(len(midi), int(test_ratio * length), replace=False)
    print(len(idx))
    for i in idx:
        shutil.move(os.path.join(ROOT_PATH, 'dataset/{}/train/{}'.format(genru,midi[i])),
                os.path.join(ROOT_PATH, 'dataset/{}/eval/{}'.format(genru,midi[i])))

shuffle_data("classic")
shuffle_data("jazz")
shuffle_data("pop")