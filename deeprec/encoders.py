import os
import random as ra
import numpy as np
import deeprec.utils.file_hd5 as hf
from sklearn.model_selection import train_test_split

class Encoders(object):
    """ """
    def __init__(self):
        pass
    
    def prepare_data(self, infile, testfile, config, test_size=0.1, sep='\t', 
                     random_state=None):
        """ """
        print('preparing data ...')
        if random_state != None:
            os.environ['PYTHONHASHSEED'] = str(random_state)
            np.random.seed(random_state)
            ra.seed(random_state)          
        seqs, resps = [], []    
        with open(infile) as f:
            for line in f:
                items = line.strip().upper().split(sep)
                seq, resp = items[0], items[1]
                seq = seq + self.__revcompl(seq)
                seqs.append(seq)
                resps.append(resp)
        seqs_test, resps_test = [], []
        with open(testfile) as f:
            for line in f:
                seq = line.strip().upper()
                seq_len = len(seq)
                seq = seq + self.__revcompl(seq)
                seqs_test.append(seq)
                resps_test.append(0)               
        seqs_train, seqs_val, resps_train, resps_val = train_test_split(
                seqs, resps, test_size=test_size, random_state=random_state)      
        encode_seqs = {'train': seqs_train, 'val': seqs_val, 'test': seqs_test}
        encode_resps = {'train': resps_train, 'val': resps_val, 
                        'test': resps_test}
        encode_maps = {'hbond_major': hbond_major_encode,
                       'hbond_minor': hbond_minor_encode}
        for k1, v1 in encode_seqs.items():
            for k2, v2 in encode_maps.items():
                outfile = infile.replace('.txt', '.'.join(['',k1,k2,'tmp']))
                with open(outfile, 'w') as f:
                    for seq in encode_seqs[k1]:
                        encode = self.__encode_sequence(seq, v2)
                        encode_string = ','.join(str(e) for e in encode)
                        f.write(encode_string + '\n')            
            seqs_trim = [item[0:seq_len] for item in encode_seqs[k1]]    
            infile_prefix = infile.replace('.txt', '.'.join(['',k1]))
            hf.ascii_to_hd5(infile_prefix, seqs_trim, encode_resps[k1])            
        
    def __revcompl(self, s):
        """ """
        rev_s = ''.join([revcompl_map[B] for B in s][::-1])
        return rev_s                
        
    def __encode_sequence(self, sequence, encode_map, reshape_length=1):
        """ """
        encode_array  = np.asarray([], dtype = int)
        for c in sequence:
            encode_array = np.append(encode_array, encode_map[c])
        return encode_array

dna_onehot_encode = {
    'A': [1,0,0,0],
    'C': [0,1,0,0],
    'G': [0,0,1,0],
    'T': [0,0,0,1],
    'N': [0,0,0,0],
    'a': [1,0,0,0],
    'c': [0,1,0,0],
    'g': [0,0,1,0],
    't': [0,0,0,1],
    'n': [0,0,0,0]
}

hbond_major_encode = {
        'A': [0,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0],
        'C': [1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1],
        'G': [0,0,0,1,0,0,0,1,0,0,1,0,1,0,0,0],
        'T': [0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,1],
        'M': [0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1],
        'g': [0,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0],        
        'N': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        'a': [0,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0], #
        'c': [1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1], #
        't': [0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,1], #
        'n': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #
        # for mismatch
        'B': [0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1], # AA
        'D': [0,0,0,1,0,0,1,0,0,0,1,0,1,0,0,0], # AC
        'E': [0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1], # AG
        'F': [1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1], # CA
        'H': [1,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0], # CC
        'I': [1,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0], # CT
        'J': [0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1], # GA
        'K': [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1], # GG
        'L': [0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0], # GT
        'O': [0,1,0,0,0,0,0,1,0,0,1,0,1,0,0,0], # TC
        'P': [0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1], # TG
        'Q': [0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,0]  # TT
    }

hbond_minor_encode = {
        'A': [0,0,0,1,1,0,0,0,0,0,0,1],
        'C': [0,0,0,1,0,0,1,0,0,0,0,1],
        'G': [0,0,0,1,0,0,1,0,0,0,0,1],
        'T': [0,0,0,1,1,0,0,0,0,0,0,1],
        'M': [0,0,0,1,0,0,1,0,0,0,0,1],
        'g': [0,0,0,1,0,0,1,0,0,0,0,1],
        'N': [0,0,0,0,0,0,0,0,0,0,0,0],
        'a': [0,0,0,1,1,0,0,0,0,0,0,1],
        'c': [0,0,0,1,0,0,1,0,0,0,0,1],
        't': [0,0,0,1,1,0,0,0,0,0,0,1],
        'n': [0,0,0,0,0,0,0,0,0,0,0,0],        
        # for mismatch
        'B': [0,0,0,1,1,0,0,0,0,0,0,1], # AA
        'D': [0,0,0,1,1,0,0,0,0,0,0,1], # AC
        'E': [0,0,0,1,0,0,1,0,0,0,0,1], # AG
        'F': [0,0,0,1,1,0,0,0,0,0,0,1], # CA
        'H': [0,0,0,1,1,0,0,0,0,0,0,1], # CC
        'I': [0,0,0,1,1,0,0,0,0,0,0,1], # CT
        'J': [0,0,0,1,0,0,1,0,0,0,0,1], # GA
        'K': [0,0,0,1,0,0,1,0,0,0,0,1], # GG
        'L': [0,0,0,1,0,0,1,0,0,0,0,1], # GT
        'O': [0,0,0,1,1,0,0,0,0,0,0,1], # TC
        'P': [0,0,0,1,0,0,1,0,0,0,0,1], # TG
        'Q': [0,0,0,1,1,0,0,0,0,0,0,1]  # TT
    }
    
revcompl_map = {
        'A':'T',
        'C':'G',
        'G':'C',
        'T':'A',
        'N':'N',
        'M':'g',
        'g':'M',
        'B':'B',
        'D':'F',
        'E':'J',
        'F':'D',
        'H':'H',
        'I':'O',
        'J':'E',
        'K':'K',
        'L':'P',
        'O':'I',
        'P':'L',
        'Q':'Q'    
    }