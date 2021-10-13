# DeepRec (<u>Deep</u> <u>Rec</u>ognition for TF–DNA binding)
DNA-binding proteins selectively bind to their genomic binding sites and trigger various cellular processes. This selective binding occurs when the DNA-binding domain of the protein recognizes its binding site by reading physicochemical signatures on the base-pair edges.


![alt text](https://github.com/TsuPeiChiu/deeprec/blob/main/deeprec/imgs/figure1.jpg)


DeepRec is a deep-learning-based method that integrates two CNN modules for extracting important physicochemical signatures in the major and minor grooves of DNA. Each CNN module extracts nonlinear spatial information between functional groups in DNA base pairs to mine potential insights beyond DNA sequence.


![alt text](https://github.com/TsuPeiChiu/deeprec/blob/main/deeprec/imgs/figure2.jpg)


# Commands to run DeepRec
./tests/deeprec_run.py -c ./max/config/config.yaml -t ./max/config/tune_params.yaml -p 3 -s 0 -e 10 -q 0.5 -d ./max/data/test_seq.txt -i ./max/data/r0_r1_max_selex_seq_10mer_150.txt -v 0.1

## Parameters 
-c: Config for training a model

-t: Config for hyperparameter search

-p: Number of param sets for hyperparameter search

-e: Number of models for ensemble learning

-q: Quantile of models selected for analysis

-s: Seed for reproducing result

-i: Input file from SELEX tool

-v: Validation set size

-d: Target sequence for interpretation

## Systems tested
- max (ACCACGTGGT)

python ./deeprec_run.py -c ./max/config/config.yaml -t ./max/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./max/data/test_seq.txt -i ./max/data/r0_r1_max_selex_seq_10mer_150.txt -v 0.1

- max_smile (ACCACGTGGT)

python ./deeprec_run.py -c ./max_smile/config/config.yaml -t ./max_smile/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./max/data/test_seq.txt -i ./max_smile/data/r0_r1_max_smile_seq_10mer_50.txt -v 0.1

- mef2b (CTAATATTAG, CTAAAAATAG)

python ./deeprec_run.py -c ./mef2b/config/config.yaml -t ./mef2b/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./max/data/test_seq.txt -i ./mef2b/data/r0_r1_mef2b_selex_seq_10mer_100.txt -v 0.1

- p53 (GGACATGTCC)

python ./deeprec_run.py -c ./p53/config/config.yaml -t ./p53/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./p53/data/test_seq.txt -i ./p53/data/r0_r1_p53_selex_seq_10mer_300.txt -v 0.1

- ATF4 (ATGACGTCAT)

python ./deeprec_run.py -c ./atf4/config/config.yaml -t ./atf4/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./atf4/data/test_seq.txt -i ./atf4/data/r0_r1_atf4_selex_seq_10mer_50.txt -v 0.1

