import os
import numpy as np
#from deprecated.sequence import EventSeq, ControlSeq
#import tensorflow as tf
import midi_processor.processor as sequence
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
            data[j] = random.randrange(3,par.event_dim+3)
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

def choice_num(pro, pre, start):
    #print(pro.shape)
    if pre == 3:
        pro = pro[4:52]
        p_idx = np.argsort(pro)[::-1] + 4
        result = int(np.random.choice(p_idx[:3], 1))
        return result, result-4
    elif pre < 52:
        pro = pro[52:68]
        p_idx = np.argsort(pro)[::-1] + 52
    elif pre < 68:
        pro = pro[68:152]
        p_idx = np.argsort(pro)[::-1] + 68
        #print(p_idx % 12)
        result = int(np.random.choice(p_idx[:12], 1))
        #print(result%12, result//12)
        return result, start
    elif pre < 152:
        pro = pro[152:168]
        p_idx = np.argsort(pro)[::-1] + 152
    elif pre < 168:
        pro = pro[3:52]
        if start > 0:
            pro[1:start+1] = 0
        p_idx = np.argsort(pro)[::-1] + 3
        if start == 47:
            result = int(np.random.choice(p_idx[:2], 1))
        else:
            result = int(np.random.choice(p_idx[:3], 1))
        #print(p_idx[:10], start+4, result)
        return result, result-4
        
    else:
        print('error')
    
    result = int(np.random.choice(p_idx[:4], 1))
    #print(p_idx[:10], pre, result)
    return result, start

if __name__ == '__main__':

    s = np.array([np.array([1,2]*50),np.array([1,2,3,4]*25)])

    t = np.array([np.array([2,3,4,5,6]*20),np.array([1,2,3,4,5]*20)])
    print(t.shape)

    print(get_masked_with_pad_tensor(100,s,t))

