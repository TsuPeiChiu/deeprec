# DeepRec (<u>Deep</u> <u>Rec</u>ognition for TF–DNA binding)
DNA-binding proteins selectively bind to their genomic binding sites and trigger various cellular processes. This selective binding occurs when the DNA-binding domain of the protein recognizes its binding site by reading physicochemical signatures on the base-pair edges.


![alt text](https://github.com/TsuPeiChiu/deeprec/blob/main/deeprec/imgs/figure1.jpg)



DeepRec is a deep-learning-based method that integrates two CNN modules for extracting important physicochemical signatures in the major and minor grooves of DNA. Each CNN module extracts nonlinear spatial information between functional groups in DNA base pairs to mine potential insights beyond DNA sequence.


![alt text](https://github.com/TsuPeiChiu/deeprec/blob/main/deeprec/imgs/figure2.jpg)


# Commands to run DeepRec
## Main command
./tests/deeprec_run.py -c ./max/config/config.yaml -t ./max/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./max/data/test_seq.txt -i ./max/data/r0_r1_max_selex_seq_10mer_150.txt -v 0.1 -f 0 -m 0124 -y 0.8


## Parameters
-c: Config for training a model

-t: Config for hyperparameter search

-p: Number of param sets for hyperparameter search

-s: Seed for reproducing result

-e: Number of models for ensemble learning

-q: Quantile of models selected for analysis

-d: Target sequence for interpretation

-i: Input file from SELEX tool

-v: Validation set size

-f: Flag for shuffling y

-m: Mode for running specific DeepRec functions (0:data generation; 1:model tuning; 2:ensemble modeling; 3:prediction with exisiting model; 4:model interpreting). Mode 2 and 3 are not allowed to coexist

-y: Y-axis limits for the plot 


## Systems tested
- max (ACCACGTGGT)

python ./deeprec_run.py -c ./max/config/config.yaml -t ./max/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./max/data/test_seq.txt -i ./max/data/r0_r1_max_selex_seq_10mer_150.txt -v 0.1 -f 0 -m 0124 -y 0.8

python ./deeprec_run.py -c ./max/config/config.yaml -t ./max/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0 -d ./max/data/test_seq.txt -i ./max/data/r0_r1_max_selex_seq_10mer_150.txt -v 0.1 -f 1 -m 24 -y 0.8 (for control)

- max_smile (ACCACGTGGT)

python ./deeprec_run.py -c ./max_smile/config/config.yaml -t ./max_smile/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./max_smile/data/test_seq.txt -i ./max_smile/data/r0_r1_max_smile_seq_10mer_50.txt -v 0.1 -f 0 -m 0124 -y 0.6

python ./deeprec_run.py -c ./max_smile/config/config.yaml -t ./max_smile/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0 -d ./max_smile/data/test_seq.txt -i ./max_smile/data/r0_r1_max_smile_seq_10mer_50.txt -v 0.1 -f 1 -m 24 -y 0.6 (for control)

- mef2b (CTAATATTAG, CTAAAAATAG)

python ./deeprec_run.py -c ./mef2b/config/config.yaml -t ./mef2b/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./mef2b/data/test_seq.txt -i ./mef2b/data/r0_r1_mef2b_selex_seq_10mer_100.txt -v 0.1 -f 0 -m 0124 -y 0.6

python ./deeprec_run.py -c ./mef2b/config/config.yaml -t ./mef2b/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0 -d ./mef2b/data/test_seq.txt -i ./mef2b/data/r0_r1_mef2b_selex_seq_10mer_100.txt -v 0.1 -f 1 -m 24 -y 0.6 (for control)


- p53 (GGACATGTCC)

python ./deeprec_run.py -c ./p53/config/config.yaml -t ./p53/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./p53/data/test_seq.txt -i ./p53/data/r0_r1_p53_selex_seq_10mer_300.txt -v 0.1 -f 0 -m 0124 -y 0.6

python ./deeprec_run.py -c ./p53/config/config.yaml -t ./p53/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0 -d ./p53/data/test_seq.txt -i ./p53/data/r0_r1_p53_selex_seq_10mer_300.txt -v 0.1 -f 1 -m 24 -y 0.6 (for control)


- atf4_episelex

python ./deeprec_run.py -c ./atf4_episelex/config/config.yaml -t ./atf4_episelex/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./atf4_episelex/data/test_seq.txt -i ./atf4_episelex/data/r0_r1_atf4_episelex_seq_10mer_75_U_M_nor.txt -v 0.1 -f 0 -m 0124 -y 0.8


- cebpb_episelex

python ./deeprec_run.py -c ./cebpb_episelex/config/config.yaml -t ./cebpb_episelex/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./cebpb_episelex/data/test_seq.txt -i ./cebpb_episelex/data/r0_r1_cebpb_episelex_seq_10mer_25_U_M_nor.txt -v 0.1 -f 0 -m 0124 -y 0.8

