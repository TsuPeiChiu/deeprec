import os
import numpy as np
import random as ra
import keras
import tensorflow as tf
import deeprec.models as dm
import deeprec.params as pr
import deeprec.metrics as me

class DeepRecEmsembler(object):
    """ """
    def __init__(self, config, nb_models, quantile=0.5, random_state=None):
        """ """
        if random_state != None:
            os.environ['PYTHONHASHSEED'] = str(random_state)
            np.random.seed(random_state)
            ra.seed(random_state)        
        self.config = config
        self.nb_models = nb_models
        self.models = []
        self.performances = []
        self.selected_models = []
        self.selected_performances = []
        self.quantile = quantile
        self.random_states = np.random.randint(10000, size=(self.nb_models))
        self.params = pr.Params(self.config)        
        self.output_path = self.params.output_path
        self.out_performs = os.path.join(self.params.output_path, 
                                         self.params.model_performances)
        self.out_test_prediction = os.path.join(self.params.output_path, 
                                                self.params.test_predictions)
        self.out_model_selected = os.path.join(self.params.output_path, 
                                               self.params.model_selected)
        
        deeprec_model = dm.DeepRecModel(self.config, random_state=random_state)
        self.train_x = deeprec_model.train_x
        self.train_y = deeprec_model.train_y
        self.train_seqs = deeprec_model.train_seqs
        self.seq_len = deeprec_model.seq_len        
        self.val_x = deeprec_model.val_x
        self.val_y  = deeprec_model.val_y
        self.val_seqs = deeprec_model.val_seqs
        self.test_x = deeprec_model.test_x
        self.test_y  = deeprec_model.test_y
        self.test_seqs = deeprec_model.test_seqs
        del(deeprec_model)


    def fit(self, verbose=False, is_shuffle=False):
        """ """
        for i in range(len(self.random_states)):
            print("fitting with seed " + str(i+1) + "/" \
                  + str(self.nb_models) + " ...")
                        
            if i == 0:
                deeprec_model = dm.DeepRecModel(self.config, 
                                            random_state=self.random_states[i])
                self.train_x = deeprec_model.train_x
                self.train_y = deeprec_model.train_y
                self.train_seqs = deeprec_model.train_seqs
                self.seq_len = deeprec_model.seq_len        
                self.val_x = deeprec_model.val_x
                self.val_y  = deeprec_model.val_y
                self.val_seqs = deeprec_model.val_seqs
                self.test_x = deeprec_model.test_x
                self.test_y  = deeprec_model.test_y
                self.test_seqs = deeprec_model.test_seqs 
             
                if is_shuffle==True:                    
                    np.random.shuffle(self.train_y)
                    np.random.shuffle(self.val_y)
                    
                    #np.random.shuffle(np.transpose(self.train_x))
                    #np.random.shuffle(np.transpose(self.val_x))
                
            else:
                input_data={'train_x':self.train_x,
                            'train_y':self.train_y,
                            'train_seqs':self.train_seqs,
                            'seq_len':self.seq_len,
                            'val_x':self.val_x,
                            'val_y':self.val_y,
                            'val_seqs':self.val_seqs,
                            'test_x':self.test_x,
                            'test_y':self.test_y,
                            'test_seqs':self.test_seqs}                                
                deeprec_model = dm.DeepRecModel(self.config, 
                                            random_state=self.random_states[i],
                                            input_data=input_data)

            self.performances.append(deeprec_model.fit(verbose=verbose))
            self.models.append(deeprec_model)
        print("resulting performance: \n" + str(self.performances))
                            
        cutoff = np.quantile(self.performances, self.quantile)
        
        with open(self.out_model_selected, 'w') as mfile:
            for performance, model in zip(self.performances, self.models):
                if performance > cutoff:
                    self.selected_models.append(model)
                    self.selected_performances.append(performance)
                    model_name = 'model.' + str(model.random_state) + '.h5'
                    model_path = os.path.join(self.output_path, model_name)
                    mfile.write(model_path + '\t')
                    model.save(model_path)
                            
        print("selected performance: \n" + str(self.selected_performances))
        print("average r-squared: " + str(np.mean(self.selected_performances)))
        
        with open(self.out_performs, 'w') as pfile:
            pfile.writelines("%f\t" % p for p in self.selected_performances)
                    
        return self.selected_models

    
    def predict(self):
        """ """
        with open(self.out_test_prediction, 'w') as tfile:
            test_pred = self.predict_average(mode=['test'])['test'] 
            for i in range(len(test_pred)):          
                tfile.write(str(self.test_seqs[i][0:self.seq_len]) + '\t' + 
                            str(test_pred[i]) + '\n')


    def predict_average(self, mode=['train','val','test']):
        """ """        
        ys = {'train':[], 'val':[], 'test':[]}
        ys_avg = {'train':None, 'val':None, 'test':None}        
        for m in mode:
            for i in range(len(self.selected_models)):
                if m=='train':
                    ys[m].append(self.selected_models[i].predict(self.train_x))
                elif m=='val':
                    ys[m].append(self.selected_models[i].predict(self.val_x))
                elif m=='test':
                    ys[m].append(self.selected_models[i].predict(self.test_x))
                         
            ys_avg[m] = np.mean(ys[m], axis=0)            
            ys_avg[m] = [val for sublist in ys_avg[m] for val in sublist]    
            
        return ys_avg


    def load_models(self):
        """ """      
        with open(self.out_model_selected, 'r') as mfile:
            path_models = mfile.readlines()[0].strip().split('\t')

        for i in path_models:
            model = keras.models.load_model(i, 
                                custom_objects={'tf': tf,
                                                'r_squared': me.r_squared})                    
            input_data={'train_x':self.train_x,
                            'train_y':self.train_y,
                            'train_seqs':self.train_seqs,
                            'seq_len':self.seq_len,
                            'val_x':self.val_x,
                            'val_y':self.val_y,
                            'val_seqs':self.val_seqs,
                            'test_x':self.test_x,
                            'test_y':self.test_y,
                            'test_seqs':self.test_seqs}
            deeprec_model = dm.DeepRecModel(self.config, 
                                            input_data=input_data)
            deeprec_model.model = model
            self.selected_models.append(deeprec_model)

        return self.selected_models
        
              