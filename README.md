This repository contains the code for our reinforcement-learning based two-stage model (RL-TSM) accepted by Transactions on Affective Computing.
\[[pdf](https://ieeexplore.ieee.org/document/9933859)\]:

>Xinhong Chen, Qing Li, Zongxi Li, Haoran Xie, Fu Lee Wang, Jianping Wang. A Reinforcement Learning Based Two-Stage Model for Emotion Cause Pair Extraction. in IEEE Transactions on Affective Computing, 2022, doi: 10.1109/TAFFC.2022.3218648.

Please cite our paper if you use this code.

## Dependencies

- **Python 3** (tested on python 3.7.13)
- [Pytorch](https://pytorch.org/) 1.10.0

## Usage

To run the program, directly run the corresponding command for different schemes. For 10-fold setting, run:
- python main.py --pretrain 1
or for 20-fold setting, run:
- python main_20fold.py --pretrain 1

If any problems, feel free to contact: xinhong.chen@cityu.edu.hk
