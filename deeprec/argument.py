import argparse as ag

class ArgumentReader(object):
    def __init__(self, args):
        p = ag.ArgumentParser(description='Run tuning and emsemble modeling')
        p.add_argument('-c', action='store', dest='config', 
                        help='Config for training a model')
        p.add_argument('-t', action='store', dest='config_tune', 
                        help='Config for hyperparameter search')
        p.add_argument('-p', action='store', type=int, dest='nb_params', 
                        help='Number of param sets for hyperparameter search')
        p.add_argument('-e', action='store', type=int, dest='nb_models', 
                        help='Number of models for ensemble learning')
        p.add_argument('-q', action='store', type=float, dest='quantile', 
                        help='Quantile of models selected for analysis')
        p.add_argument('-s', action='store', type=int, dest='random_state', 
                        help='Seed for reproducing result')
        p.add_argument('-i', action='store', dest='input_selex', 
                        help='Input file from SELEX tool')
        p.add_argument('-v', action='store', type=float, dest='valid_size', 
                        help='Validation set size')
        p.add_argument('-d', action='store', dest='target_seq', 
                        help='Target sequence for interpretation')
        p.add_argument('-f', action='store', type=int, dest='shuffle', 
                       default=0, help='Flag for shuffling y')
        
        args = p.parse_args()         
        self.config = args.config
        self.config_tune = args.config_tune
        self.nb_params = args.nb_params
        self.nb_models = args.nb_models
        self.quantile = args.quantile
        self.random_state = args.random_state
        self.input_selex = args.input_selex
        self.valid_size = args.valid_size
        self.target_seq = args.target_seq        
        self.shuffle=True if args.shuffle==1 else False 
