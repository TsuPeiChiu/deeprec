import pandas as pd
#import deeprec.models as dm

def remove_redundancy(input_selex):
    """ """
    print('removing redundancy ...')
    df = pd.read_csv(input_selex, sep='\t')
    outfile = input_selex.replace('.txt', '.onestrand.txt')
    with open(outfile, 'w') as f: 
        kmer, saff = '', '' 
        for idx, row in df.iterrows():
            if kmer!=reverse_compl(row['Kmer']):
                kmer = row['Kmer']
                saff = row['SymmetrizedAffinity']
                f.write(kmer + '\t' + str(saff) + '\n')           
            else:            
                kmer = row['Kmer']
                saff = row['SymmetrizedAffinity']
    return outfile

def reverse_compl(seq):
    alt_map = {'ins':'0'}
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'M':'g', 'g':'M'}

    for k,v in alt_map.items():
        seq = seq.replace(k,v)
    bases = list(seq)
    bases = reversed([complement.get(base,base) for base in bases])
    bases = ''.join(bases)
    for k,v in alt_map.items():
        bases = bases.replace(v,k)
    return bases

"""
def find_index(config, seq):
    """ """
    deeprec_model = dm.DeepRecModel(config=config)
    data_type = ['train', 'val']
    target_seq, target_type, target_index = seq, None, None
    
    for t in data_type:
        if t=='train':
            for i in range(len(deeprec_model.train_seqs)):
                if deeprec_model.train_seqs[i].astype(str)==seq:
                    target_type, target_index = t, i
                    break
        else:
            for i in range(len(deeprec_model.val_seqs)):
                if deeprec_model.val_seqs[i].astype(str)==seq:
                    target_type, target_index = t, i
                    break                                
    print('target_seq: ' + target_seq)
    print('target_type: ' + target_type)
    print('target_index: ' + str(target_index))
    return target_type, target_index
"""