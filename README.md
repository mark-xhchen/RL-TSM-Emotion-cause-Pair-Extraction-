This repository contains the code for our reinforcement-learning based two-stage model (RL-TSM) accepted by Transactions on Affective Computing.
%\[[pdf](https://www.aclweb.org/anthology/2020.coling-main.18.pdf)\]:

>Xinhong Chen, Qing Li, Zongxi Li, Haoran Xie, Fu Lee Wang, Jianping Wang. A Reinforcement Learning Based Two-Stage Model for Emotion Cause Pair Extraction. In Transactions on Affective Computing.

Please cite our paper if you use this code.

## Dependencies

- **Python 3** (tested on python 3.7.13)
- [Pytorch](https://pytorch.org/) 1.10.0

## Usage

To run the program, directly run the corresponding command for different schemes. For 10-fold setting, run:
- python main.py --pretrain 1
or for 20-fold setting, run:
- python main_20fold.py --pretrain 1

If any problems, feel free to contact: xinhong.chen@my.cityu.edu.hk
