groove_map = {'major':4, 'minor':3}
groove_names = {'major':'M', 'minor':'m'}
channel_map = {
    'A': '[0, 0, 0, 1]',
    'D': '[0, 0, 1, 0]',
    'M': '[0, 1, 0, 0]',
    'N': '[1, 0, 0, 0]'
}
signature_map = {
    'major':{
        0:{'A':['A','G'], 'D':[], 'M':['T'], 'N':['C']},
        1:{'A':['T','G'], 'D':['A','C'], 'M':[], 'N':[]},
        2:{'A':['A','C'], 'D':['T','G'], 'M':[], 'N':[]},
        3:{'A':['T','C'], 'D':[], 'M':['A'], 'N':['G']}
    },  
    'minor':{
        0:{'A':['A','T', 'G', 'C'], 'D':[], 'M':[], 'N':[]},
        1:{'A':[], 'D':['G','C'], 'M':[], 'N':['A','T']},
        2:{'A':['A','T', 'G', 'C'], 'D':[], 'M':[], 'N':[]}
    }
}
dispaly_map = {
    'major':{
        0:{'A':['A'], 'C':['N','M'], 'G':['A'], 'T':['M']},
        1:{'A':['D'], 'C':['D'], 'G':['A'], 'T':['A']},
        2:{'A':['A'], 'C':['A'], 'G':['D'], 'T':['D']},
        3:{'A':['M'], 'C':['A'], 'G':['N','M'], 'T':['A']}
    },  
    'minor':{
        0:{'A':['A'], 'C':['A'], 'G':['A'], 'T':['A']},
        1:{'A':['N'], 'C':['D'], 'G':['D'], 'T':['N']},
        2:{'A':['A'], 'C':['A'], 'G':['A'], 'T':['A']}
    }
}
results_column = ['seq', 'type', 'h_pos',
                       's_pos', 'channel', 'delta', 'sem']
seq_letters = ['A','C','G','T']
seq_letters_rev = ['T','G','C','A']
pc_letters = ['N','M','D','A']
results_tune_column = ['trail_idx','cross_val_idx','params',
                       'loss','val_loss','r_squared','val_r_squared']

def generate_names_1d(seq_len):
    """"""
    names = []
    seq_idx = range(1,seq_len+1)
    pc_idx = list(range(1,5)) + list(range(1,4)) 
    M_idx = ['M']*4
    m_idx = ['m']*3
    sig_idx = ['N', 'M', 'D', 'A']
    for i in sig_idx:
        for k in range(1,5):
                for l in seq_idx:
                    name = str(l)
                    names.append(name+"_M_"+str(k)+"_"+i)
        for k in range(1,4):
                for l in seq_idx:
                    name = str(l)
                    names.append(name+"_m_"+str(k)+"_"+i)
    return names 
                     