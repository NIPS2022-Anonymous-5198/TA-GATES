# TA-GATES
## Introduction

Dear Reviewers, this is the cleaned code for submitting "TA-GATES: An Encoding Scheme for Neural Network Architectures".

The training scripts are put under `scripts/`, and the implementation of TA-GATES is put under `my_nas/`. Besides, we also provide the training data (under `data/`) and configurations (under `cfgs/`) to enable the reproducibility. Some training logs and checkpoints are placed under `results/`. In the following parts, we will guide you to reproduce the experiments in our paper.

To install all requirements, run
```sh
$ conda create -n mynas python==3.7.6 pip
$ conda activate mynas
$ pip install -r requirements.txt
```

## Experiments on NAS-Bench-101

### Preparation
Install the NAS-Bench-101 package following instructions [here](https://github.com/google-research/nasbench).

### Run Experiments
To evaluate the performance of TA-GATES with a training proportion 10%, one should run the following instructions:
```sh
$ python scripts/train_nasbench101.py cfgs/nb101_cfgs/tagates.yaml --gpu 0 --seed [random seed] --train-dir results/nb101_tagates_tr1e-1/ --save-every 200 --eval-only-last 5 --train-pkl data/nasbench-101/nasbench101_train_1.pkl --valid-pkl data/nasbench-101/nasbench101_valid.pkl --train-ratio 0.1
```

To evaluate the anytime performance of TA-GATES with a training proportion 10%, one should run the following instructions:
```sh
$ python scripts/train_nasbench101_anytime.py cfgs/nb101_cfgs/tagates_anytime.yaml --gpu 0 --seed [random seed] --train-dir results/nb101_tagates_anytime_tr1e-1/ --save-every 200 --eval-only-last 5 --train-pkl data/nasbench-101/nasbench101_train_anytime_1.pkl --valid-pkl data/nasbench-101/nasbench101_valid_anytime.pkl --train-ratio 0.1
```

## Evaluation on NAS-Bench-301

### Preparation
Install the NAS-Bench-301 package following instructions [here](https://github.com/automl/nasbench301).

### Run Evaluation
To evaluate the performance of TA-GATES with a training proportion 10%, one should run the following instructions:
```sh
$ python scripts/train_nasbench301.py cfgs/nb301_cfgs/tagates.yaml --gpu 0 --seed [random seed] --train-dir results/nb301_tagates_tr1e-1/ --save-every 200 --eval-only-last 5 --train-pkl data/nasbench-301/nasbench301_train_mtx_1.pkl --valid-pkl data/nasbench-301/nasbench301_valid_mtx.pkl --train-ratio 0.1
```

To evaluate the anytime performance of TA-GATES with a training proportion 10%, one should run the following instructions:
```sh
$ python scripts/train_nasbench301_anytime.py cfgs/nb301_cfgs/tagates_anytime.yaml --gpu 0 --seed [random seed] --train-dir results/nb301_tagates_anytime_tr1e-1/ --save-every  200 --eval-only-last 5 --train-pkl data/nasbench-301/nasbench301_train_mtx_anytime_1.pkl --valid-pkl data/nasbench-301/nasbench301_valid_mtx_anytime.pkl --train-ratio 0.1
```

## License
The codes of NAS-Bench-101 and NAS-Bench-301 are licensed Apache 2.0, and the code of NAS-Bench-201 is licensed MIT.
