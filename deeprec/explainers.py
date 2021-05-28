import os
import numpy as np
import random as ra
import pandas as pd
import math as ma
import deeprec.names as na
import deeprec.params as pr
import deeprec.visualizers as vi
import lime.lime_tabular
#from sklearn import linear_model

class DeepRecExplainer(object):
    """"""
    def __init__(self, config, models, x, seq, random_state=None):        
        """"""
        if random_state != None:
            os.environ['PYTHONHASHSEED'] = str(random_state)
            np.random.seed(random_state)
            ra.seed(random_state)            
        self.models = models
        self.x = x
        self.seq = seq
        self.random_state = random_state
        self.params = pr.Params(config, self.random_state)
        self.seq_len = int(len(self.x)/28) # with padding
        self.pad_len = self.seq_len-2*len(self.seq)
        self.names_1d = na.generate_names_1d(self.seq_len)
        self.groove_map = na.groove_map
        self.groove_names = na.groove_names
        self.channel_map = na.channel_map
        self.seq_map = na.seq_map
        self.dispaly_map = na.dispaly_map
        self.results_column = na.results_column        
        self.seq_letters = na.seq_letters
        self.seq_letters_rev = na.seq_letters_rev
        self.pc_letters = na.pc_letters
        self.groove_map = na.groove_map
    
    def plot_logos(self, outfile=None):
        """"""      
        self.samples, self.samples_name = self.__perturb()
        self.ys = self.__predict()
        results = self.__calculate_logos()        
        logos_file = self.params.model_logos if outfile is None else outfile
        outfile = os.path.join(self.params.output_path, logos_file)               
        vi.plot_logos(outfile, self.seq, results)
        
    def __perturb(self):
        """"""
        samples, samples_name = [], []
        p = self.x.copy()
        s = self.seq.copy()
        samples.append(p)
        samples_name.append(s)              
        for i in range(len(self.seq)):
            i_rev = self.seq_len-i-1     
            for j, seq_letter in enumerate(self.seq_letters):                  
                for g_type, nb_pos in self.groove_map.items():                        
                    for l in range(nb_pos):                            
                        # nullify the pc
                        p_null = self.x.copy()
                        s_null = self.seq + '_'.join(['',str(i+1),
                                                self.groove_names[g_type],
                                                str(l+1),'null'])
                        for pc_letter in self.pc_letters:
                            key = '_'.join([str(i+1),
                                            self.groove_names[g_type],
                                            str(l+1),pc_letter])
                            idx = self.names_1d.index(key)                            
                            key_rev = '_'.join([str(i_rev+1),
                                            self.groove_names[g_type],
                                            str(nb_pos-l),pc_letter])
                            idx_rev = self.names_1d.index(key_rev)
                            p_null[idx] = 0
                            p_null[idx_rev] = 0                                
                        samples.append(p_null)
                        samples_name.append(s_null)

                        # add A,D,M,N to null
                        for c in na.pc_letters:
                            p_add = p_null.copy()
                            s_add = '_'.join([s_null,c])
                            key = '_'.join([str(i+1),
                                        self.groove_names[g_type],
                                        str(l+1),c])
                            idx = self.names_1d.index(key)
                            key_rev = '_'.join([str(i_rev+1),
                                                self.groove_names[g_type],
                                                str(nb_pos-l),c])
                            idx_rev = self.names_1d.index(key_rev)
                            p_add[idx] = 1
                            p_add[idx_rev] = 1
                            samples.append(p_add)
                            samples_name.append(s_add)
#        print(samples_name)                              
        return samples, samples_name

    def __predict(self):
        """"""
        ys = []
        for model in self.models:
            y = model.predict(np.array(self.samples))
            ys.append(y)
        return ys

    def __calculate_logos(self):
        """"""
        results = pd.DataFrame(columns=self.results_column)         
        for s_pos in range(len(self.seq)):
            for g_type, nb_pos in self.groove_map.items():
                for h_pos in range(nb_pos):
                    for pc_type, pc_code in self.channel_map.items():                                                                     
                        if pc_type in self.dispaly_map[g_type][h_pos] \
                                                        [self.seq[s_pos]]:                           
                            if pc_type in self.seq_map[g_type] \
                                                        [self.seq[s_pos]] \
                                                        [h_pos]:                                                                                                                        
                                diffs_means, diffs_sems = [], []
                                key_ref = '_'.join([self.seq, 
                                                    str(s_pos+1),
                                                    self.groove_names[g_type],
                                                    str(h_pos+1),'null',
                                                    pc_type])
                                key_null = '_'.join([self.seq, 
                                                     str(s_pos+1),
                                                     self.groove_names[g_type],
                                                     str(h_pos+1),'null'])                                
                            else:
                                diffs_means, diffs_sems = [], []
                                seq_pc_type = self.seq_map[g_type] \
                                                [self.seq[s_pos]] \
                                                [h_pos]                                
                                key_ref = '_'.join([self.seq, 
                                                     str(s_pos+1),
                                                     self.groove_names[g_type],
                                                     str(h_pos+1), 'null',
                                                     pc_type])
                                key_null = '_'.join([self.seq, 
                                                    str(s_pos+1),
                                                    self.groove_names[g_type],
                                                    str(h_pos+1), 'null',
                                                    seq_pc_type])                                                              
                            diffs_mean, diffs_sem = \
                                self.__calculate_diffs(key_ref, key_null)                                                                        
                            diffs_means.append(diffs_mean)
                            diffs_sems.append(diffs_sem)                                                                                                                                                   
                            pc_mean, pc_sem = self.__calculate_mean_sem(
                                    diffs_means, diffs_sems)
                            results = results.append({'seq': self.seq,
                                    'type': g_type, 
                                    'h_pos': h_pos, 
                                    's_pos': s_pos, 
                                    'channel': self.channel_map[pc_type], 
                                    'delta': pc_mean,
                                    'sem': pc_sem}, ignore_index=True)
                                                                                                                                               
        return results
    
    def __calculate_diffs(self, key_ref, key_null):
        """
        difference between models
        """
        idx_ref = self.samples_name.index(key_ref)
        idx_null = self.samples_name.index(key_null)     
        diffs = []
        for i in range(len(self.ys)):
            val_ref = self.ys[i][idx_ref]
            val_null = self.ys[i][idx_null]            
            if val_null!=0:
                dddG = ma.log(val_ref)-ma.log(val_null)
            else:
                dddG = ma.log(val_ref)-ma.log(0.000001)            
            diffs.append(dddG)
        
        """       
        if 1==1:
        #if key_null.find('_M_2_')!=-1:
            print(key_ref)
            print(key_null)
            print(self.ys[0][idx_ref])
            print(self.ys[0][idx_null])
            print('')
        """          
            
        diffs_mean = np.mean(diffs)
        diffs_std = np.std(diffs)
        diffs_sem = diffs_std/np.sqrt(len(diffs))        
        return diffs_mean, diffs_sem
    
    def __calculate_mean_sem(self, diffs_means, diffs_sems):
        pc_mean = np.mean(diffs_means)        
        pc_sem = np.sqrt(np.sum(np.square(diffs_sems)))
        return pc_mean, pc_sem
            
    

    


        
class DeepRecLimeExplainer(object):
    """ """            
    def __init__(self, models, x, seq, nb_features, nb_samples, seed=0):
        """ """
        self.models = models
        self.x = x
        self.seq = seq
        self.seq_len = int(len(self.x)/28) # with padding
        self.names_1d = na.generate_names_1d(self.seq_len)
        self.results_column = na.results_column
        self.groove_map = na.groove_map
        self.groove_names = na.groove_names
        self.channel_map = na.channel_map
        self.lime_outputs = []
                
        for model in models:
            explainer = lime.lime_tabular.LimeTabularExplainer(model.train_x,
                                                               verbose=True,
                                                               mode='regression',
                                                               feature_names=self.names_1d,
                                                               categorical_features=list(range(len(self.x))),
                                                               feature_selection='lasso_path',
                                                               random_state=seed)                        
            ins = explainer.explain_instance(x, 
                                             model.predict, 
                                             num_features=nb_features, 
                                             num_samples=nb_samples, 
                                             model_regressor=None, 
                                             sampling_method='deeprec',
                                             seq_len=len(self.seq), 
                                             max_mutations=2)                        
            items = ins.as_map()[1]
            result = {}
            for item in items:
                result[item[0]] = item[1]           
            self.lime_outputs.append(result) 

            print('r-squared:' + str(ins.score[0]))
                            
    def plot_logos(self):
        """"""
        results = pd.DataFrame(columns=self.results_column)
                
        for s_pos in range(len(self.seq)):
            for g_type, nb_pos in self.groove_map.items():
                for h_pos in range(nb_pos):
                    for pc_type, pc_code in self.channel_map.items():
                        key = '_'.join([str(s_pos+1),self.groove_names[g_type],str(h_pos+1), pc_type])
                        key_rev = '_'.join([str(self.seq_len-s_pos),
                                            self.groove_names[g_type],
                                            str(nb_pos-h_pos), 
                                            pc_type])                        
                        idx = self.names_1d.index(key)
                        idx_rev = self.names_1d.index(key_rev)                        
                        value, value_rev = 0, 0
                        if idx in self.lime_outputs[0]:
                            value = float(self.lime_outputs[0][idx])                            
                        if idx_rev in self.lime_outputs[0]:
                            value_rev = float(self.lime_outputs[0][idx_rev])
                            
                        score = np.mean([value, value_rev])
                        
                        channel_map = {
                        'A': '[0, 0, 0, 1]', 
                        'D': '[0, 0, 1, 0]',
                        'M': '[0, 1, 0, 0]',
                        'N': '[1, 0, 0, 0]'
                        }
                        channel = channel_map[pc_type]
                        sem = 0
                        
                        results = results.append({'seq': self.seq,
                                      'type': g_type, 
                                      'h_pos': h_pos, 
                                      's_pos': s_pos, 
                                      'channel': channel, 
                                      'delta': score,
                                      'sem': sem}, ignore_index=True)
                        
        vi.plot_logos(self.seq, results)