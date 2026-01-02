import sys
from scipy import stats
import numpy as np
import random
import torch

class Logger(object):
    def __init__(self, fileN="Default.logs"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.log.flush()
        
        
def conditional_samples(e):
    anchor_e = e[0]
    gid = [0]
    
    for k in range(1, e.shape[0]):
        if stats.pearsonr(e[0], e[k])[0] > 0:
            gid += [0]
        else:
            gid += [1]
            
    return gid

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def print_results(res):
    p_eval = ''
    count = 0
    for keys, values in res.items():
        p_eval += keys + ':' + '[%.4f]' % round(values, 4) + ' '
        count += 1
        if count == 4:
            print(p_eval)
            p_eval = ''
            count = 0