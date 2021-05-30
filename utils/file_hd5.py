import os, h5py
import numpy as np

def ascii_to_hd5(infile_prefix, seqs, resps):
    """ """
    outfile = '.'.join([infile_prefix,'h5'])
    with h5py.File(outfile, 'w') as fp:    
        data_group = fp.create_group('data')
       
        # X
        encode_types = {'hbond_major':4, 'hbond_minor':3}
        for k, v in encode_types.items():
            infile_tmp = '.'.join([infile_prefix, k, 'tmp'])
            data = np.genfromtxt(infile_tmp, delimiter= ',') ###

            width = int(data.shape[1]/(v*4))
            dataset = data.reshape(data.shape[0], 4, v, width, order='F') 
            dataset_name = '_'.join([k, 'x'])
            hdf = data_group.create_dataset(dataset_name, dataset.shape, 
                                            dtype = dataset.dtype)                                                                           
            hdf[...] = dataset
            os.remove(infile_tmp)
                    
        # Y
        dataset_name = 'c0_y'
        resps = np.array(resps).astype(float).reshape(len(resps), 1)
        hdf = data_group.create_dataset(dataset_name, resps.shape, 
                                        dtype='float')
        hdf[...] = resps

        # extra information
        dataset_name = 'probe_seq'               
        seqs = np.array(seqs).reshape(len(seqs))
#        seqs = np.string_(seqs).reshape(len(seqs), 1)
        hdf = data_group.create_dataset(dataset_name, seqs.shape, 
                                        dtype=seqs.dtype)
        hdf[...] = seqs
