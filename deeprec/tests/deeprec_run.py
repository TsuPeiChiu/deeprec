import sys
import deeprec.encoders as ec
import deeprec.models as dm
import deeprec.ensembler as eb
import deeprec.explainers as ep
import deeprec.utils.file_selex as fs
from deeprec.argument import ArgumentReader

def main(args):
    """ """
    ar = ArgumentReader(args)
        
    # 0. data preparation
    onestrand_selex = fs.remove_redundancy(ar.input_selex)
    deeprec_encoder = ec.Encoders()
    deeprec_encoder.prepare_data(infile=onestrand_selex, 
                                 config=ar.config, 
                                 test_size=ar.valid_size, 
                                 random_state=ar.random_state)

    # 1. model tuning
    deeprec_model = dm.DeepRecModel(config=ar.config, 
                                    random_state=ar.random_state)
    config_tuned = deeprec_model.tune(config_tune=ar.config_tune, 
                       nb_params=ar.nb_params)
    
    # 2. ensemble modeling
    config_tuned = ar.config.replace('.yaml','.tuned.yaml')
    deeprec_emsembler = eb.DeepRecEmsembler(config=config_tuned, 
                                            nb_models=ar.nb_models, 
                                            quantile=ar.quantile, 
                                            random_state=ar.random_state)
    deeprec_models = deeprec_emsembler.fit(verbose=False)
    
    # 3. model interpreting
    seq_type, seq_idx = fs.find_index(config_tuned, ar.target_seq)
    deeprec_explainer = ep.DeepRecExplainer(config=config_tuned, 
                        models=deeprec_models,
                        x=deeprec_models[0].train_x[seq_idx], 
                        seq=deeprec_models[0].train_seqs[seq_idx].astype(str),
                        random_state=ar.random_state)
    deeprec_explainer.plot_logos()
    
    
if __name__=='__main__':
    main(sys.argv)
