# deeprec running command
./tests/deeprec_run.py -c ./max/config/config.yaml -t ./max/config/tune_params.yaml -p 3 -s 0 -e 10 -q 0.5 -d ./max/data/test_seq.txt -i ./max/data/r0_r1_max_selex_seq_10mer_150.txt -v 0.1

# Parameters 
-c: Config for training a model

-t: Config for hyperparameter search

-p: Number of param sets for hyperparameter search

-e: Number of models for ensemble learning

-q: Quantile of models selected for analysis

-s: Seed for reproducing result

-i: Input file from SELEX tool

-v: Validation set size

-d: Target sequence for interpretation

# Systems tested
- max (ACCACGTGGT)

python ./deeprec_run.py -c ./max/config/config.yaml -t ./max/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./max/data/test_seq.txt -i ./max/data/r0_r1_max_selex_seq_10mer_150.txt -v 0.1

- max_smile (ACCACGTGGT)

python ./deeprec_run.py -c ./max_smile/config/config.yaml -t ./max_smile/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./max/data/test_seq.txt -i ./max_smile/data/r0_r1_max_smile_seq_10mer_50.txt -v 0.1

- mef2b (CTAATATTAG, CTAAAAATAG)

python ./deeprec_run.py -c ./mef2b/config/config.yaml -t ./mef2b/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./max/data/test_seq.txt -i ./mef2b/data/r0_r1_mef2b_selex_seq_10mer_100.txt -v 0.1

- p53 (GGACATGTCC)

python ./deeprec_run.py -c ./p53/config/config.yaml -t ./p53/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./p53/data/test_seq.txt -i ./p53/data/r0_r1_p53_selex_seq_10mer_300.txt -v 0.1

- ATF4 (ATGACGTCAT)

export CUDA_VISIBLE_DEVICES=3
python ./deeprec_run.py -c ./atf4/config/config.yaml -t ./atf4/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./atf4/data/test_seq.txt -i ./atf4/data/r0_r1_atf4_selex_seq_10mer_50.txt -v 0.1

