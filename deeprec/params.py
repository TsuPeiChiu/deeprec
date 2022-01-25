import os
import yaml
import itertools
import random as ra
import numpy as np

class Params(object):
    """ """    
    def __init__(self, config, random_state=None):
        if random_state != None:
            os.environ['PYTHONHASHSEED']=str(random_state)
            np.random.seed(random_state)
            ra.seed(random_state)
        self.random_state = random_state
        self.config = config
        
        with open(self.config) as file:
            c = yaml.full_load(file)            
            self.train = c['input']['train']
            self.val = c['input']['val']
            self.test = c['input']['test']
            self.output_path = c['output']['path']
            self.model_tune = c['output']['model_tune']
            self.model_selected = c['output']['model_selected']
            self.model_logos = c['output']['model_logos']
            self.model_logos_results = c['output']['model_logos_results']
            self.model_performances = c['output']['model_performances']
            self.test_predictions = c['output']['test_predictions']  
            self.optimizer = c['optimizer']
            self.optimizer_params = {
                'lr': float(c['optimizer_params']['lr'])
            }
            self.loss = c['loss']
            self.nb_epoch = c['nb_epoch']
            self.batch_size = c['batch_size']  
            self.hbond_major = {
                'nb_filter_1': int(c['cartridges']['hbond_major']['nb_filter'][0]),
                'filter_len_1': int(c['cartridges']['hbond_major']['filter_len'][0]),
                'filter_hei_1': int(c['cartridges']['hbond_major']['filter_hei'][0]),
                'pool_len_1': int(c['cartridges']['hbond_major']['pool_len'][0]),
                'pool_hei_1': int(c['cartridges']['hbond_major']['pool_hei'][0]),
                'activation_1': str(c['cartridges']['hbond_major']['activation'][0]),                                
                'l1_1': float(c['cartridges']['hbond_major']['l1'][0]),
                'l2_1': float(c['cartridges']['hbond_major']['l2'][0]),

                'nb_filter_2': int(c['cartridges']['hbond_major']['nb_filter'][1]),
                'filter_len_2': int(c['cartridges']['hbond_major']['filter_len'][1]),
                'filter_hei_2': int(c['cartridges']['hbond_major']['filter_hei'][1]),
                'pool_len_2': int(c['cartridges']['hbond_major']['pool_len'][1]),
                'pool_hei_2': int(c['cartridges']['hbond_major']['pool_hei'][1]),
                'activation_2': str(c['cartridges']['hbond_major']['activation'][1]),                                
                'l1_2': float(c['cartridges']['hbond_major']['l1'][1]),
                'l2_2': float(c['cartridges']['hbond_major']['l2'][1])          
            }            
            self.hbond_minor = {
                'nb_filter_1': int(c['cartridges']['hbond_minor']['nb_filter'][0]),
                'filter_len_1': int(c['cartridges']['hbond_minor']['filter_len'][0]),
                'filter_hei_1': int(c['cartridges']['hbond_minor']['filter_hei'][0]),
                'pool_len_1': int(c['cartridges']['hbond_minor']['pool_len'][0]),
                'pool_hei_1': int(c['cartridges']['hbond_minor']['pool_hei'][0]),
                'activation_1': str(c['cartridges']['hbond_minor']['activation'][0]),                                
                'l1_1': float(c['cartridges']['hbond_minor']['l1'][0]),
                'l2_1': float(c['cartridges']['hbond_minor']['l2'][0]),
                
                'nb_filter_2': int(c['cartridges']['hbond_minor']['nb_filter'][1]),
                'filter_len_2': int(c['cartridges']['hbond_minor']['filter_len'][1]),
                'filter_hei_2': int(c['cartridges']['hbond_minor']['filter_hei'][1]),
                'pool_len_2': int(c['cartridges']['hbond_minor']['pool_len'][1]),
                'pool_hei_2': int(c['cartridges']['hbond_minor']['pool_hei'][1]),
                'activation_2': str(c['cartridges']['hbond_minor']['activation'][1]),                                
                'l1_2': float(c['cartridges']['hbond_minor']['l1'][1]),
                'l2_2': float(c['cartridges']['hbond_minor']['l2'][1])
            }            
            self.joint = {
                'nb_hidden': int(c['joint']['nb_hidden']),
                'activation': str(c['joint']['activation']),
                'l1': float(c['joint']['l1']),
                'l2': float(c['joint']['l2']),
                'drop_out': int(c['joint']['drop_out'])
            }            
            self.target = {
                'activation': str(c['target']['activation'])
            }

    def get_grid(self, config_tune, nb_params=50):
        """ """
        with open(config_tune) as f:
            c = yaml.full_load(f)            
            g, lr, nb_epoch, batch_size = [], [], [], []
            cartridges_l1_1, cartridges_l2_1 = [], []
            cartridges_l1_2, cartridges_l2_2 = [], []
            cartridges_filter_len_1, cartridges_filter_len_2 = [], []
            cartridges_nb_filter_1, cartridges_nb_filter_2 = [], []
            joint_nb_hidden, joint_l1, joint_drop_out = [], [], []
            
            for i in c['lr']: lr.append({'lr':i})
            for i in c['nb_epoch']: nb_epoch.append({'nb_epoch':i})
            for i in c['batch_size']: batch_size.append({'batch_size':i})
            for i in c['cartridges_l1_1']: 
                cartridges_l1_1.append({'cartridges_l1_1':i})
            for i in c['cartridges_l1_2']: 
                cartridges_l1_2.append({'cartridges_l1_2':i})
            for i in c['cartridges_l2_1']: 
                cartridges_l2_1.append({'cartridges_l2_1':i})
            for i in c['cartridges_l2_2']: 
                cartridges_l2_2.append({'cartridges_l2_2':i})
            for i in c['cartridges_filter_len_1']:    
                cartridges_filter_len_1.append({'cartridges_filter_len_1':i})  
            for i in c['cartridges_filter_len_2']:    
                cartridges_filter_len_2.append({'cartridges_filter_len_2':i})
            for i in c['cartridges_nb_filter_1']:    
                cartridges_nb_filter_1.append({'cartridges_nb_filter_1':i}) 
            for i in c['cartridges_nb_filter_2']:    
                cartridges_nb_filter_2.append({'cartridges_nb_filter_2':i})                               
            for i in c['joint_nb_hidden']: 
                joint_nb_hidden.append({'joint_nb_hidden':i})
            for i in c['joint_l1']: 
                joint_l1.append({'joint_l1':i})
            for i in c['joint_drop_out']: 
                joint_drop_out.append({'joint_drop_out':i})
                                
            if c['lr']!=[]: g+=[lr]
            if c['nb_epoch']!=[]: g+=[nb_epoch]
            if c['batch_size']!=[]: g+=[batch_size]
            if c['cartridges_l1_1']!=[]: g+=[cartridges_l1_1]
            if c['cartridges_l1_2']!=[]: g+=[cartridges_l1_2]                                                
            if c['cartridges_l2_1']!=[]: g+=[cartridges_l2_1]
            if c['cartridges_l2_2']!=[]: g+=[cartridges_l2_2]            
            if c['cartridges_filter_len_1']!=[]: g+=[cartridges_filter_len_1]
            if c['cartridges_filter_len_2']!=[]: g+=[cartridges_filter_len_2]            
            if c['cartridges_nb_filter_1']!=[]: g+=[cartridges_nb_filter_1]
            if c['cartridges_nb_filter_2']!=[]: g+=[cartridges_nb_filter_2]
            if c['joint_l1']!=[]: g+=[joint_l1]
            if c['joint_drop_out']!=[]: g+=[joint_drop_out]
            if c['joint_nb_hidden']!=[]: g+=[joint_nb_hidden]
            
            all_comb = list(itertools.product(*g))
            selected_comb = []
            idx = ra.sample(range(len(all_comb)), nb_params)
            for i in idx: 
                selected_comb.append(all_comb[i])
        return selected_comb       
        
    def update(self, item):
        """ """
        for i in item:
            for k, v in i.items():
                if k=='lr': 
                    self.optimizer_params['lr']=float(i['lr'])
                if k=='nb_epoch': 
                    self.nb_epoch=int(i['nb_epoch'])
                if k=='batch_size': 
                    self.batch_size=int(i['batch_size'])
                if k=='cartridges_l1_1': 
                    self.hbond_major['l1_1']=float(i['cartridges_l1_1'])
                    self.hbond_minor['l1_1']=float(i['cartridges_l1_1'])
                if k=='cartridges_l1_2': 
                    self.hbond_major['l1_2']=float(i['cartridges_l1_2'])
                    self.hbond_minor['l1_2']=float(i['cartridges_l1_2'])
                if k=='cartridges_l2_1': 
                    self.hbond_major['l2_1']=float(i['cartridges_l2_1'])
                    self.hbond_minor['l2_1']=float(i['cartridges_l2_1'])
                if k=='cartridges_l2_2': 
                    self.hbond_major['l2_2']=float(i['cartridges_l2_2'])
                    self.hbond_minor['l2_2']=float(i['cartridges_l2_2'])                                                                               
                if k=='cartridges_filter_len_1': 
                    self.hbond_major['filter_len_1']= \
                            int(i['cartridges_filter_len_1'])
                    self.hbond_minor['filter_len_1']= \
                            int(i['cartridges_filter_len_1'])                              
                if k=='cartridges_filter_len_2': 
                    self.hbond_major['filter_len_2']= \
                            int(i['cartridges_filter_len_2'])
                    self.hbond_minor['filter_len_2']= \
                            int(i['cartridges_filter_len_2'])
                if k=='cartridges_nb_filter_1': 
                    self.hbond_major['nb_filter_1']= \
                            int(i['cartridges_nb_filter_1'])
                    self.hbond_minor['nb_filter_1']= \
                            int(i['cartridges_nb_filter_1'])                       
                if k=='cartridges_nb_filter_2': 
                    self.hbond_major['nb_filter_2']= \
                            int(i['cartridges_nb_filter_2'])
                    self.hbond_minor['nb_filter_2']= \
                            int(i['cartridges_nb_filter_2'])      
                if k=='joint_nb_hidden': 
                    self.joint['nb_hidden']=int(i['joint_nb_hidden'])
                if k=='joint_l1': 
                    self.joint['l1']=float(i['joint_l1'])
                if k=='joint_drop_out': 
                    self.joint['drop_out']=float(i['joint_drop_out'])
    
    def save(self, outfile):
        """ """
        c = {}
        c['input'] = {'train': self.train, 'val': self.val, 'test': self.test}
        c['output'] = {'path': self.output_path,
                         'model_tune': self.model_tune,
                         'model_selected': self.model_selected, 
                         'model_logos': self.model_logos,
                         'model_logos_results': self.model_logos_results,
                         'model_performances': self.model_performances,
                         'test_predictions': self.test_predictions}
        c['optimizer'] = self.optimizer
        c['optimizer_params'] = {'lr': self.optimizer_params['lr']}
        c['loss'] = self.loss
        c['nb_epoch'] = self.nb_epoch
        c['batch_size'] = self.batch_size 
        c['cartridges'] = {
                'hbond_major':{
                        'nb_filter': [self.hbond_major['nb_filter_1'],
                                      self.hbond_major['nb_filter_2']],
                        'filter_len': [self.hbond_major['filter_len_1'],
                                       self.hbond_major['filter_len_2']],                       
                        'filter_hei': [self.hbond_major['filter_hei_1'],
                                         self.hbond_major['filter_hei_2']],
                        'pool_len': [self.hbond_major['pool_len_1'],
                                     self.hbond_major['pool_len_2']],
                        'pool_hei': [self.hbond_major['pool_hei_1'],
                                       self.hbond_major['pool_hei_2']],
                        'activation': [self.hbond_major['activation_1'],
                                         self.hbond_major['activation_2']],
                        'l1': [self.hbond_major['l1_1'],
                               self.hbond_major['l1_2']],
                        'l2': [self.hbond_major['l2_1'],
                               self.hbond_major['l2_2']]},
                                                                        
                'hbond_minor':{
                        'nb_filter': [self.hbond_minor['nb_filter_1'],
                                      self.hbond_minor['nb_filter_2']],
                        'filter_len': [self.hbond_minor['filter_len_1'],
                                       self.hbond_minor['filter_len_2']],
                        'filter_hei': [self.hbond_minor['filter_hei_1'],
                                       self.hbond_minor['filter_hei_2']],
                        'pool_len': [self.hbond_minor['pool_len_1'],
                                     self.hbond_minor['pool_len_2']],
                        'pool_hei': [self.hbond_minor['pool_hei_1'],
                                     self.hbond_minor['pool_hei_2']],
                        'activation': [self.hbond_minor['activation_1'],
                                       self.hbond_minor['activation_2']],
                        'l1': [self.hbond_minor['l1_1'],
                               self.hbond_minor['l1_2']],
                        'l2': [self.hbond_minor['l2_1'],
                               self.hbond_minor['l2_2']]}} 
                       
        c['joint'] = {'nb_hidden': self.joint['nb_hidden'],
                         'activation': self.joint['activation'],
                         'l1': self.joint['l1'],
                         'l2': self.joint['l2'],
                         'drop_out': self.joint['drop_out']}
        c['target'] = {'activation': self.target['activation']}
        
        with open(outfile, 'w') as f:
            yaml.dump(c, f)
          