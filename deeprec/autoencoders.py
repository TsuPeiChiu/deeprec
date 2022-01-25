import os, gc, math
import numpy as np
import random as ra
import tensorflow as tf
import tensorflow.keras.callbacks as cb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import deeprec.params as pr
import deeprec.nets as ne
import deeprec.names as na
import deeprec.metrics as me
import deeprec.utils.file_utils as fut
from keras import backend as K


class DeepRecAutoencoder(object):
    """ """
    def __init__(self, config, h5_file=None, 
                 random_state=None, input_data=None):
        """ """
        if random_state != None:
            os.environ['PYTHONHASHSEED'] = str(random_state)
            np.random.seed(random_state)
            ra.seed(random_state)

        self.random_state = random_state
        self.config = config
        self.params = pr.Params(self.config, self.random_state)
        if not os.path.exists(self.params.output_path):
            os.mkdir(self.params.output_path)
            
        self.__prepare_input()        
        self.__build(h5_file)
        
        
    def fit(self, is_tune=False, idx_params_tune=None, verbose=True, is_shuffle=False):
        """ """
        callback_early_stopping = cb.EarlyStopping(monitor='loss', patience=5)
        def lr_exp_decay(epoch, lr):
            return self.params.optimizer_params['lr']*math.exp(-0.01*epoch)
        
        
        

        self.model.fit(self.train_x,
                                    self.train_x,
                                    epochs=self.params.nb_epoch)
                        

            
    def __build(self, h5_file=None):
        """ """
        self.model = ne.build_autoencoder_model(self.params, 
                                                self.seq_len,
                                                self.random_state)
    
    def predict(self):
        """ """
        print (self.test_x.shape)
        
        self.test_x[0][1][0][4] = 1
        self.test_x[0][1][3][19] = 1
        
        
        pred = self.model.predict(self.test_x)
#        print(pred)
 
        
        list_inputs = []
        for i in range(4):
            tmp_input = np.zeros((4, 24))
            for s_pos in range(0,24):
                for h_pos in range(0,4): 
                    tmp_input[h_pos][s_pos] = self.test_x[0][i][h_pos][s_pos]
            list_inputs.append(tmp_input)


        list_results = []
        for i in range(4):
            tmp_result = np.zeros((4, 24))
            for s_pos in range(0,24):
                for h_pos in range(0,4): 
                    tmp_result[h_pos][s_pos] = pred[0][i][h_pos][s_pos]
            list_results.append(tmp_result)



        """
        
        results = np.zeros((4, 24))
        for i in range(4):       
            
            for s_pos in range(0,24):
                for h_pos in range(0,4): 
                    results[h_pos][s_pos] = self.test_x[0][i][h_pos][s_pos]
                    
#            print(results)
        
        results = np.zeros((4, 24))
        for i in range(4):
            for s_pos in range(0,24):
                for h_pos in range(0,4): 
                    results[h_pos][s_pos] = pred[0][i][h_pos][s_pos]
                    
#            print(results)
            print(results.shape)            


        for s_pos in range(0,24):
            for h_pos in range(0,4): 
                results[h_pos][s_pos] = pred[0][1][h_pos][s_pos]

        """
        import seaborn as sns
        
        h_names = ['H-bond acceptor', 'H-bond donor', 'Methyl group', 'Nonpolar hydrogen']
        h_colors = ['Reds', 'Blues', 'YlOrBr', 'Greys']
        
        f, ax = plt.subplots(4, 2, figsize=(12,9))
        for i in range(4):
            cut_input = list_inputs[i][:,0:10]
            cut_result = list_results[i][:,0:10]
            
            sns.heatmap(cut_input, ax=ax[3-i,0], square=True, linewidths=2, cbar_kws={"shrink": .4}, cmap=h_colors[3-i], annot=True, annot_kws={"fontsize":7})
            sns.heatmap(cut_result, ax=ax[3-i,1], square=True, linewidths=2, cbar_kws={"shrink": .4}, cmap=h_colors[3-i], annot=True, annot_kws={"fontsize":7})
            ax[3-i,0].set_ylabel(h_names[3-i])
            ax[3-i,1].set_ylabel(h_names[3-i])
            
            ax[3-i,0].set_yticklabels(['FG4','FG3','FG2','FG1'])
            ax[3-i,1].set_yticklabels(['FG4','FG3','FG2','FG1'])
        
        ax[0,0].set_xticklabels(['A','C','C','A','C','G','T','G','G','T'])
        ax[0,0].xaxis.set_label_position('top')
        ax[0,0].xaxis.tick_top()
        
        ax[0,1].set_xticklabels(['A','C','C','A','C','G','T','G','G','T'])
        ax[0,1].xaxis.set_label_position('top')
        ax[0,1].xaxis.tick_top()
        
        
        ax[3,0].set_xticklabels(['T','G','G','T','G','C','A','C','C','A'])        
        ax[3,1].set_xticklabels(['T','G','G','T','G','C','A','C','C','A']) 
        ax[2,0].set_xticklabels([])        
        ax[2,1].set_xticklabels([]) 
        ax[1,0].set_xticklabels([])        
        ax[1,1].set_xticklabels([]) 

        
        f.tight_layout()
        
        
        
        """
        a = results[:,0:10]
        
        print(a.shape)
        
        
        ax = sns.heatmap(a, square=True, linewidths=2, cbar_kws={"shrink": .4}, cmap='Reds', annot=True, annot_kws={"fontsize":7})
        ax.set_yticklabels(['FG4','FG3','FG2','FG1'])
        ax.set_xticklabels(['A','C','C','A','C','G','T','G','G','T'])
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        """
        
        
        plt.savefig('abc', bbox_inches='tight') 
        
#        return self.model.predict(x)
    
    
    

    def __prepare_input(self):
        """ """
        # train data
        train_file, train_data = fut.read_hdf(self.params.train, 1024)        
        train_x_major = np.array(train_data['hbond_major_x']) 
        train_x_minor = np.array(train_data['hbond_minor_x'])      


        #train_x_con = np.concatenate([train_x_major, train_x_minor], axis=2)
        train_x_con = train_x_major

        patch_len = self.params.hbond_major['filter_len']
        patch_col = np.zeros(train_x_con.shape[0]*
                             train_x_con.shape[1]*
                             train_x_con.shape[2]*
                             patch_len)
        patch_col = patch_col.reshape(train_x_con.shape[0],
                             train_x_con.shape[1],
                             train_x_con.shape[2],
                             patch_len)
        seq_len = int(train_x_con.shape[-1]/2)
        train_x_patched = np.concatenate((train_x_con[:,:,:,:seq_len], 
                                          patch_col, 
                                          train_x_con[:,:,:,seq_len:]), axis=3)
        self.train_x = train_x_patched

        self.train_y = np.array(train_data['c0_y'])
        self.train_seqs = train_data['probe_seq']
        self.seq_len = seq_len
        
        # val data
        val_file, val_data = fut.read_hdf(self.params.val, 1024)
        val_x_major = np.array(val_data['hbond_major_x'])
        val_x_minor = np.array(val_data['hbond_minor_x'])
#        val_x_con = np.concatenate([val_x_major, val_x_minor], axis=2)
        val_x_con = val_x_major
        patch_len = self.params.hbond_minor['filter_len']
        patch_col = np.zeros(val_x_con.shape[0]*
                             val_x_con.shape[1]*
                             val_x_con.shape[2]*
                             patch_len)
        patch_col = patch_col.reshape(val_x_con.shape[0],
                             val_x_con.shape[1],
                             val_x_con.shape[2],
                             patch_len)
        seq_len = int(val_x_con.shape[-1]/2)
        val_x_patched = np.concatenate((val_x_con[:,:,:,:seq_len], 
                                        patch_col, 
                                        val_x_con[:,:,:,seq_len:]), axis=3)              
        self.val_x = val_x_patched
        self.val_y  = np.array(val_data['c0_y'])
        self.val_seqs = val_data['probe_seq']
        
        # test data
        test_file, test_data = fut.read_hdf(self.params.test, 1024)
        test_x_major = np.array(test_data['hbond_major_x'])
        test_x_minor = np.array(test_data['hbond_minor_x'])
#        test_x_con = np.concatenate([test_x_major, test_x_minor], axis=2)
        test_x_con = test_x_major
        patch_len = self.params.hbond_minor['filter_len']
        patch_col = np.zeros(test_x_con.shape[0]*
                             test_x_con.shape[1]*
                             test_x_con.shape[2]*
                             patch_len)
        patch_col = patch_col.reshape(test_x_con.shape[0],
                             test_x_con.shape[1],
                             test_x_con.shape[2],
                             patch_len)
        seq_len = int(test_x_con.shape[-1]/2)
        test_x_patched = np.concatenate((test_x_con[:,:,:,:seq_len], 
                                        patch_col, 
                                        test_x_con[:,:,:,seq_len:]), axis=3)              
        self.test_x = test_x_patched
        self.test_y  = np.array(test_data['c0_y'])
        self.test_seqs = test_data['probe_seq']
        