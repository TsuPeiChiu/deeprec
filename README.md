# deeprec running command
deeprec_run.py -c ./max/config/config.yaml -t ./max/config/tune_params.yaml -p 3 -s 0 -e 10 -q 0.5 -d ACCACGTGGT -i ./max/data/r0_r1_max_selex_seq_10mer_150.txt -v 0.1

 
-c: Config for training a model
-t: Config for hyperparameter search
-p: Number of param sets for hyperparameter search
-e: Number of models for ensemble learning
-q: Number of models for ensemble learning
-s: Seed for reproducing result
-i: Input file from SELEX tool
-v: Validation set size
-d: Target sequence for interpretation