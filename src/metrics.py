from __future__ import print_function
from __future__ import division
import sys
def roc_curve(true, pred, pos_label=1):
    assert len(true) == len(pred) and len(true) > 0
    thresholds = sorted(pred, reverse=True)
    tpr = []
    fpr = []
    size = len(true)
    P = sum([ele==pos_label for ele in true])
    N = size - P
    for thres in thresholds:
        tp = 0
        fp = 0
        for idx in range(size):
            if pred[idx] >= thres:
                if true[idx] == pos_label:
                    tp += 1
                else:
                    fp += 1
        tpr.append(1.0 * tp/P)
        fpr.append(1.0 * fp/N)
    return tpr, fpr, thresholds

def roc_auc_score(true, pred, pos_label=1):
    tpr, fpr, thresholds = roc_curve(true, pred, pos_label)
    if sys.version_info > (3,0,0):
        import matplotlib.pyplot as plt
        plt.plot(fpr, tpr)
        plt.xlabel('fpr')
        plt.ylabel('tpr')
        plt.title('result in auroc')
        plt.savefig('auroc.png')
    auroc_score = 0
    size = len(fpr)
    if size <=1:
        return 1
    for idx in range(1,size):
        auroc_score += (tpr[idx-1] + tpr[idx]) * (fpr[idx] - fpr[idx-1]) / 2
    return auroc_score

if __name__ == "__main__":
    true = [0, 0, 1, 1, 0, 1, 1]
    pred = [0.1, 0.4, 0.35, 0.8, 0.42, 0.73, 0.1]
    score = roc_auc_score(true, pred)
    print("%.4f" % score)
