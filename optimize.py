from inspect import getmembers
import numpy as np
from pkg_resources import get_entry_map
import torch
from metrics import calc_metrics, calcF1


def calcReward(emo_tags, cau_tags, gt_emotags, gt_cautags):
    r = [0 for _ in range(len(emo_tags))]
    ridx = 0
    for i in range(len(emo_tags)):
        if emo_tags[i] == gt_emotags[i]:
            r[i] += 1
            if emo_tags[i] == 1:
                for j in range(len(emo_tags)):
                    if cau_tags[ridx][j] == gt_cautags[i][j]:
                        r[i] += 1 
                    else:
                        r[i] -= 1 
                ridx += 1
        elif emo_tags[i] == 1:
            r[i] -= 1
            for j in range(len(emo_tags)):
                if cau_tags[ridx][j] > 0:
                    r[i] -= 1 
            ridx += 1
        else:
            r[i] -= 1

    return r


def calcFinalReward(emo_tags, cau_tags, gt_emotags, gt_cautags):
    made_actions = len(emo_tags)

    r = -1.
    tp, fp, fn = calc_metrics(emo_tags, cau_tags, gt_emotags, gt_cautags)
    if tp[-1] + fp[-1] != 0 and tp[-1] + fn[-1] != 0:
        _, _, r = calcF1(tp[-1], fp[-1], fn[-1])
    
    r *= made_actions

    return r
    

def calcGrad(emo_tags, emo_actprobs, cau_actprobs, step_reward, final_reward, pretrain, device=torch.device("cpu")):
    length = len(emo_tags)
    decay_reward = final_reward
    ridx = 0 
    grads = torch.FloatTensor(1, ).fill_(0).to(device)
    for i in range(length)[::-1]:
        decay_reward = decay_reward * 0.95 + step_reward[i]
        to_grad = -torch.log(emo_actprobs[i]).to(device)
        if emo_tags[i] > 0:
            for j in range(length):
                to_grad -= torch.log(cau_actprobs[ridx][j]).to(device)
            ridx += 1

        if not pretrain:
            to_grad *= torch.FloatTensor(1, ).fill_(decay_reward).to(device)

        grads = grads + to_grad
    return grads


def optimize(emo_tags, emo_actprobs, cau_tags, cau_actprobs, gt_emotags, gt_cautags, pretrain, device):
    step_reward = calcReward(emo_tags, cau_tags, gt_emotags, gt_cautags)
    final_reward = calcFinalReward(emo_tags, cau_tags, gt_emotags, gt_cautags)
    grads = calcGrad(emo_tags, emo_actprobs, cau_actprobs, step_reward, final_reward, pretrain, device)
    loss = grads.cpu().data[0]
    grads.backward()
    
    return loss


def optimize_round(sample_emo_tags, sample_emo_actprobs, sample_cau_tags, sample_cau_actprobs, gt_emotags, gt_cautags, pretrain, device):
    sample_round = len(sample_emo_tags)
    loss = 0.
    # real optimisation with top/bot biases taken into account
    for i in range(sample_round):
        loss += optimize(sample_emo_tags[i], sample_emo_actprobs[i], sample_cau_tags[i], sample_cau_actprobs[i], gt_emotags, gt_cautags, pretrain, device)
    return loss / sample_round
