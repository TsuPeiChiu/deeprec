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
            self.output_path = c['output']['path']
            self.model_tune = c['output']['model_tune']
            self.model_logos = c['output']['model_logos']
            self.optimizer = c['optimizer']
            self.optimizer_params = {
                'lr': float(c['optimizer_params']['lr'])
            }
            self.loss = c['loss']
            self.nb_epoch = c['nb_epoch']
            self.batch_size = c['batch_size']            
            self.hbond_major = {
                'nb_filter': int(c['cartridges']['hbond_major']['nb_filter']),
                'filter_len': int(c['cartridges']['hbond_major']['filter_len']),
                'filter_hei': int(c['cartridges']['hbond_major']['filter_hei']),
                'pool_len': int(c['cartridges']['hbond_major']['pool_len']),
                'pool_hei': int(c['cartridges']['hbond_major']['pool_hei']),
                'activation': str(c['cartridges']['hbond_major']['activation']),                                
                'l1': float(c['cartridges']['hbond_major']['l1']),
                'l2': float(c['cartridges']['hbond_major']['l2']),
                'nb_hidden': int(c['cartridges']['hbond_major']['nb_hidden'])
            }            
            self.hbond_minor = {
                'nb_filter': int(c['cartridges']['hbond_minor']['nb_filter']),
                'filter_len': int(c['cartridges']['hbond_minor']['filter_len']),
                'filter_hei': int(c['cartridges']['hbond_minor']['filter_hei']),
                'pool_len': int(c['cartridges']['hbond_minor']['pool_len']),
                'pool_hei': int(c['cartridges']['hbond_minor']['pool_hei']),
                'activation': str(c['cartridges']['hbond_minor']['activation']),                                
                'l1': float(c['cartridges']['hbond_minor']['l1']),
                'l2': float(c['cartridges']['hbond_minor']['l2']),
                'nb_hidden': int(c['cartridges']['hbond_minor']['nb_hidden'])
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
            cartridges_l1, cartridges_nb_hidden = [], []
            joint_nb_hidden, joint_drop_out = [], []        
            for i in c['lr']: lr.append({'lr':i})
            for i in c['nb_epoch']: nb_epoch.append({'nb_epoch':i})
            for i in c['batch_size']: batch_size.append({'batch_size':i})
            for i in c['cartridges_l1']: 
                cartridges_l1.append({'cartridges_l1':i})
            for i in c['cartridges_nb_hidden']: 
                cartridges_nb_hidden.append({'cartridges_nb_hidden':i})
            for i in c['joint_nb_hidden']: 
                joint_nb_hidden.append({'joint_nb_hidden':i})
            for i in c['joint_drop_out']: 
                joint_drop_out.append({'joint_drop_out':i})        
            if c['lr']!=[]: g+=[lr]
            if c['nb_epoch']!=[]: g+=[nb_epoch]
            if c['batch_size']!=[]: g+=[batch_size]
            if c['cartridges_l1']!=[]: g+=[cartridges_l1]
            if c['cartridges_nb_hidden']!=[]: g+=[cartridges_nb_hidden]
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
                if k=='cartridges_l1': 
                    self.hbond_major['l1']=float(i['cartridges_l1'])
                    self.hbond_minor['l1']=float(i['cartridges_l1'])                
                if k=='cartridges_nb_hidden': 
                    self.hbond_major['nb_hidden']=int(i['cartridges_nb_hidden'])
                    self.hbond_minor['nb_hidden']=int(i['cartridges_nb_hidden'])
                if k=='joint_nb_hidden': 
                    self.joint['nb_hidden']=int(i['joint_nb_hidden'])
                if k=='joint_drop_out': 
                    self.joint['drop_out']=float(i['joint_drop_out'])
    
    def save(self, outfile):
        """ """
        c = {}
        c['input'] = {'train': self.train, 'val': self.val}
        c['output'] = {'path': self.output_path,
                         'model_tune': self.model_tune, 
                         'model_logos': self.model_logos}
        c['optimizer'] = self.optimizer
        c['optimizer_params'] = {'lr': self.optimizer_params['lr']}
        c['loss'] = self.loss
        c['nb_epoch'] = self.nb_epoch
        c['batch_size'] = self.batch_size 
        c['cartridges'] = {
                'hbond_major':{
                        'nb_filter': self.hbond_major['nb_filter'],
                        'filter_len': self.hbond_major['filter_len'],
                        'filter_hei': self.hbond_major['filter_hei'],
                        'pool_len': self.hbond_major['pool_len'],
                        'pool_hei': self.hbond_major['pool_hei'],
                        'activation': self.hbond_major['activation'],
                        'l1': self.hbond_major['l1'],
                        'l2': self.hbond_major['l2'],
                        'nb_hidden': self.hbond_major['nb_hidden']},
                'hbond_minor':{
                        'nb_filter': self.hbond_minor['nb_filter'],
                        'filter_len': self.hbond_minor['filter_len'],
                        'filter_hei': self.hbond_minor['filter_hei'],
                        'pool_len': self.hbond_minor['pool_len'],
                        'pool_hei': self.hbond_minor['pool_hei'],
                        'activation': self.hbond_minor['activation'],
                        'l1': self.hbond_minor['l1'],
                        'l2': self.hbond_minor['l2'],
                        'nb_hidden': self.hbond_minor['nb_hidden']}}                        
        c['joint'] = {'nb_hidden': self.joint['nb_hidden'],
                         'activation': self.joint['activation'],
                         'l1': self.joint['l1'],
                         'l2': self.joint['l2'],
                         'drop_out': self.joint['drop_out']}
        c['target'] = {'activation': self.target['activation']}
        
        with open(outfile, 'w') as f:
            yaml.dump(c, f)
          