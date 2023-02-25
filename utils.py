import os
import numpy as np
#from deprecated.sequence import EventSeq, ControlSeq
#import tensorflow as tf
import params as par
import random

def bar_id(x):
    #print(x.shape)
    bar_list = []
    for b in range(x.shape[0]):
        batch_list = []
        num = 0
        for w in range(x.shape[1]):
            if x[b][w] == 3:
                num += 1
                batch_list.append(num)
            else:
                batch_list.append(num)
        bar_list.append(batch_list)
    return np.array(bar_list)

def data_mask(data):
    length = len(data)
    per = int(length * 0.15)
    arr = np.random.randint(0, length, (per))
    for j in arr:
        n = random.randrange(100)
        if n < 10:
            continue
        elif n < 20:
            data[j] = random.randrange(3,100)
        else:
            data[j] = par.mask_token
    return data

def find_files_by_extensions(root, exts=[]):
    def _has_ext(name):
        if not exts:
            return True
        name = name.lower()
        for ext in exts:
            if name.endswith(ext):
                return True
        return False
    for path, _, files in os.walk(root):
        for name in files:
            if _has_ext(name):
                yield os.path.join(path, name)

def choice_num(pro,pre):

    if pre == 3:
        pro = pro[4:52]
        p_idx = np.argsort(pro)[::-1] + 4
    elif pre < 52:
        pro = pro[52:68]
        p_idx = np.argsort(pro)[::-1] + 52
    elif pre < 68:
        pro = pro[68:164]
        p_idx = np.argsort(pro)[::-1] + 68
    elif pre < 164:
        pro = pro[164:180]
        p_idx = np.argsort(pro)[::-1] + 164
    elif pre < 180:
        pro = pro[3:52]
        p_idx = np.argsort(pro)[::-1] + 3
        #print(p_idx[:10])
    else:
        print('error')
    
    #print(p_idx[:5])
    result = int(np.random.choice(p_idx[:5], 1))

    return result

if __name__ == '__main__':

    s = np.array([np.array([1,2]*50),np.array([1,2,3,4]*25)])

    t = np.array([np.array([2,3,4,5,6]*20),np.array([1,2,3,4,5]*20)])
    print(t.shape)

    print(get_masked_with_pad_tensor(100,s,t))

