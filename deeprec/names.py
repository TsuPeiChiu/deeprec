groove_map = {'major':4, 'minor':3}
groove_names = {'major':'M', 'minor':'m'}
channel_map = {
    'A': '[0, 0, 0, 1]',
    'D': '[0, 0, 1, 0]',
    'M': '[0, 1, 0, 0]',
    'N': '[1, 0, 0, 0]'
}

seq_map = {
    'major':{'A':{0:'A',1:'D',2:'A',3:'M'},
             'T':{0:'M',1:'A',2:'D',3:'A'},
             'C':{0:'N',1:'D',2:'A',3:'A'},        
             'G':{0:'A',1:'A',2:'D',3:'N'},
             'M':{0:'M',1:'D',2:'A',3:'A'},
             'g':{0:'A',1:'A',2:'D',3:'M'}
    },
    'minor':{'A':{0:'A',1:'N',2:'A'},
             'T':{0:'A',1:'N',2:'A'},
             'C':{0:'A',1:'D',2:'A'},
             'G':{0:'A',1:'D',2:'A'},
             'M':{0:'A',1:'D',2:'A'},
             'g':{0:'A',1:'D',2:'A'}
    }
}
    
dispaly_map = {
    'major':{
        0:{'A':['A'], 'C':['N','M'], 'G':['A'], 'T':['M'], 'M':['M'], 'g':['A']},
        1:{'A':['D'], 'C':['D'], 'G':['A'], 'T':['A'], 'M':['D'], 'g':['A']},
        2:{'A':['A'], 'C':['A'], 'G':['D'], 'T':['D'], 'M':['A'], 'g':['D']},
        3:{'A':['M'], 'C':['A'], 'G':['N','M'], 'T':['A'], 'M':['A'], 'g':['M']}
    },  
    'minor':{
        0:{'A':['A'], 'C':['A'], 'G':['A'], 'T':['A'], 'M':['A'], 'g':['A']},
        1:{'A':['N','D'], 'C':['D','N'], 'G':['D','N'], 'T':['N','D'], 'M':['D','N'], 'g':['D','N']},
        2:{'A':['A'], 'C':['A'], 'G':['A'], 'T':['A'], 'M':['A'], 'g':['A']}
    }
}
    
    
results_column = ['seq', 'type', 'h_pos',
                       's_pos', 'channel', 'delta', 'sem']
seq_letters = ['A','C','G','T', 'M', 'g']
seq_letters_rev = ['T','G','C','A', 'g', 'M']
pc_letters = ['N','M','D','A']
results_tune_column = ['trail_idx','cross_val_idx','params',
                       'loss','val_loss','r_squared','val_r_squared']

def generate_names_1d(seq_len):
    """"""
    names = []
    seq_idx = range(1,seq_len+1)
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
