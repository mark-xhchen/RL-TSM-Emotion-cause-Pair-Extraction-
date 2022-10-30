import numpy as np


def calcF1(tp, fp, fn, beta=1.0):
    """
    Get F1 score.
    """
    if tp + fp == 0 or tp + fn == 0:
        return 0, 0, 0
    precision = float(tp) / float(tp+fp)
    recall = float(tp) / float(tp+fn)
    if precision + recall < 1e-5:
        return 0, 0, 0
    return precision, recall, (1+beta*beta) * precision * recall / (beta*beta*precision + recall)


def calc_metrics(emo_tags, cause_tags, gt_emotags, gt_cautags):
    tp, fp, fn = [np.array([0, 0, 0]) for _ in range(3)]

    if cause_tags is None or len(cause_tags) == 0:
        for i in range(len(gt_emotags)):
            if gt_emotags[i] == 1 and emo_tags[i] == 1:
                tp[0] += 1
            elif gt_emotags[i] == 0 and emo_tags[i] == 1:
                fp[0] += 1
            elif gt_emotags[i] == 1 and emo_tags[i] == 0:
                fn[0] += 1
    else:
        pdj = 0
        for i in range(len(emo_tags)):
            if gt_emotags[i] == emo_tags[i] and emo_tags[i] > 0:
                tp[0] += 1
                for j in range(len(gt_cautags[i])):
                    if cause_tags[pdj][j] == 1 and gt_cautags[i][j] == 1:
                        tp[2] += 1
                    elif cause_tags[pdj][j] == 1 and gt_cautags[i][j] == 0:
                        fp[2] += 1
                    elif cause_tags[pdj][j] == 0 and gt_cautags[i][j] == 1:
                        fn[2] += 1
                pdj += 1
            elif gt_emotags[i] == 0 and emo_tags[i] > 0:
                fp[0] += 1
                fp[2] += sum(cause_tags[pdj])
                pdj += 1
            elif gt_emotags[i] > 0 and emo_tags[i] == 0:
                fn[0] += 1
                fn[2] += sum(gt_cautags[i])
        
        cau_predlabel = np.array([0 for _ in range(len(gt_emotags))])
        cau_truelabel = np.array([0 for _ in range(len(gt_emotags))])
        for cau_tag in cause_tags:
            cau_predlabel += np.array(cau_tag)
        for key in gt_cautags:
            cau_truelabel += np.array(gt_cautags[key])
        
        cau_predlabel = (cau_predlabel>0).astype(np.int32)
        cau_truelabel = (cau_truelabel>0).astype(np.int32)

        tp[1] += (cau_predlabel & cau_truelabel).astype(np.int32).sum()
        fp[1] += (cau_predlabel - cau_truelabel > 0).astype(np.int32).sum()
        fn[1] += (cau_truelabel - cau_predlabel > 0).astype(np.int32).sum()

    return tp, fp, fn