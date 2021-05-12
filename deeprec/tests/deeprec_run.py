import sys
import deeprec.models as dm
import deeprec.ensembler as eb
import deeprec.explainers as ep
from deeprec.argument import ArgumentReader

def main(args):
    """ """
    ar = ArgumentReader(args)
    
    # 1. model tuning
    deeprec_model = dm.DeepRecModel(config=ar.config, 
                                    random_state=ar.random_state)
    config_tuned = deeprec_model.tune(config_tune=ar.config_tune, 
                       nb_params=ar.nb_params)
    
    # 2. ensemble modeling   
    deeprec_emsembler = eb.DeepRecEmsembler(config=config_tuned, 
                                            nb_models=ar.nb_models, 
                                            quantile=ar.quantile, 
                                            random_state=ar.random_state)
    deeprec_models = deeprec_emsembler.fit(verbose=False)
    
    # 3. model interpreting
    deeprec_explainer = ep.DeepRecExplainer(config=config_tuned, 
                        models = deeprec_models,
                        x = deeprec_models[0].train_x[9579], #ACCACGTGGT
                        seq = deeprec_models[0].train_seqs[9579].astype(str),
                        random_state=ar.random_state)
    deeprec_explainer.plot_logos()
    

if __name__=='__main__':
    main(sys.argv)
