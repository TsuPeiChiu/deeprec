import os
import numpy as np
import random as ra
import deeprec.models as dm

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
            
    def fit(self, verbose=False):
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
                self.val_seqs =deeprec_model.val_seqs
                
            else:
                input_data={'train_x':self.train_x,
                            'train_y':self.train_y,
                            'train_seqs':self.train_seqs,
                            'seq_len':self.seq_len,
                            'val_x':self.val_x,
                            'val_y':self.val_y,
                            'val_seqs':self.val_seqs}                                
                deeprec_model = dm.DeepRecModel(self.config, 
                                            random_state=self.random_states[i],
                                            input_data=input_data)
            
            
            self.performances.append(deeprec_model.fit(verbose=verbose))
            self.models.append(deeprec_model)
        print("resulting performance: \n" + str(self.performances))
                            
        cutoff = np.quantile(self.performances, self.quantile)
        for performance, model in zip(self.performances, self.models):
            if performance > cutoff:
                self.selected_models.append(model)
                self.selected_performances.append(performance)
        
        print("selected performance: \n" + str(self.selected_performances))
        print("average r-squared: " + str(np.mean(self.selected_performances)))
        
        return self.selected_models
    
    
    def predict_average(self):
        """ """
        y_train = []
        y_val = []
        for i in range(len(self.selected_models)):
            y_train.append(self.selected_models[i].predict(self.train_x))
            y_val.append(self.selected_models[i].predict(self.val_x))
            
        y_train_avg = np.mean(y_train, axis=0)
        y_val_avg = np.mean(y_train, axis=0)
            
        return y_train_avg, y_val_avg
        