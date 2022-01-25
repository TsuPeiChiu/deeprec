import sys
import deeprec.encoders as ec
import deeprec.models as dm
import deeprec.ensembler as eb
import deeprec.explainers as ep
import deeprec.utils.file_selex as fs
from deeprec.argument import ArgumentReader

def main(args):     
    ar = ArgumentReader(args)
    mode = ar.mode #'01234'

    # 0: data preparation 
    if '0' in mode:             
        onestrand_selex = fs.remove_redundancy(ar.input_selex)
        deeprec_encoder = ec.Encoders()
        deeprec_encoder.prepare_data(infile=onestrand_selex,
                                     testfile=ar.target_seq,
                                     config=ar.config, 
                                     test_size=ar.valid_size, 
                                     random_state=ar.random_state)
    # 1: model tuning
    if '1' in mode:
        deeprec_model = dm.DeepRecModel(config=ar.config, 
                                        random_state=ar.random_state)
        config_tuned = deeprec_model.tune(config_tune=ar.config_tune, 
                           nb_params=ar.nb_params)
    # 2: ensemble modeling
    if '2' in mode:
        config_tuned = ar.config.replace('.yaml','.tuned.yaml')
        deeprec_emsembler = eb.DeepRecEmsembler(config=config_tuned, 
                                                nb_models=ar.nb_models, 
                                                quantile=ar.quantile, 
                                                random_state=ar.random_state)
        # 3: prediction from turned models
        if '3' in mode:
            deeprec_models = deeprec_emsembler.load_models()
            deeprec_emsembler.predict()
        else:
            deeprec_models = deeprec_emsembler.fit(verbose=False, 
                                                   is_shuffle=ar.shuffle)
    # 4. model interpreting
    if '4' in mode:
        deeprec_explainer = ep.DeepRecExplainer(config=config_tuned, 
                            models=deeprec_models,                        
                            xs=deeprec_models[0].test_x,
                            seqs=deeprec_models[0].test_seqs,
                            random_state=ar.random_state)
        deeprec_explainer.plot_logos()
    
    
if __name__=='__main__':
    main(sys.argv)
