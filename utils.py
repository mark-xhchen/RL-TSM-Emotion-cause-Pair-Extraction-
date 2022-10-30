import time
import numpy as np
import torch.optim as optim
from metrics import calc_metrics
from optimize import optimize_round


def run_for_batch(optimizer, model, datas, sample_round, device, pretrain, test):
    """
    Get model outputs and train model.
    """
    all_tp, all_fp, all_fn = [np.array([0, 0, 0]) for _ in range(3)]
    loss = .0
    # here, "data" is the "instance" defined in my dataloader.py
    for data in datas:
        sample_emo_tags, sample_emo_actprobs, sample_cau_tags, sample_cau_actprobs = [[] for _ in range(4)]
        if pretrain and not test:
            pre_emotags, pre_cautags = data['emo_tags'], data['cau_tags']
            pre_cautags_full = [[] for _ in range(len(pre_emotags))]
            for key in pre_cautags:
                pre_cautags_full[key] = pre_cautags[key]

        for i in range(sample_round):
            # pretraining with all given
            if pretrain and not test:
                emo_tags, emo_actprobs, cau_tags, cau_actprobs = model(test, data['texts'], pre_emotags, pre_cautags_full, device)
            # train from scratch
            else:
                emo_tags, emo_actprobs, cau_tags, cau_actprobs = model(test, data['texts'], None, None, device)

            sample_emo_tags.append(emo_tags)
            sample_emo_actprobs.append(emo_actprobs)
            sample_cau_tags.append(cau_tags)
            sample_cau_actprobs.append(cau_actprobs)

            tp, fp, fn = calc_metrics(emo_tags, cau_tags, data['emo_tags'], data['cau_tags'])
            all_tp += tp
            all_fp += fp
            all_fn += fn
            
        # training optimization
        if not test:
            optimizer.zero_grad()
            loss += optimize_round(sample_emo_tags, sample_emo_actprobs, sample_cau_tags, sample_cau_actprobs, data['emo_tags'], data['cau_tags'], pretrain, device)
            optimizer.step()

    if len(datas) == 0:
        return all_tp, all_fp, all_fn, 0
    return all_tp, all_fp, all_fn, loss / len(datas)
