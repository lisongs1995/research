import os
BASE_DIR = os.path.abspath(__file__)
import sys
sys.path.append(BASE_DIR)

import numpy as np
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_curve, auc
from src.new_loop import LocalOutlierProbability as loop
import pdb

_Debug = False

def fetch(dataset='http', fetch_percent_10=True):
    if _Debug == True:
        pdb.set_trace()
    raw_data = fetch_kddcup99(subset=dataset, percent10=fetch_percent_10)
   # unique_data, index = np.unique(raw_data.data.astype(float), axis=0, return_index=True)
   # scaler_data = StandardScaler().fit_transform(raw_data.data[index])
    scaler_data = StandardScaler().fit_transform(raw_data.data)
    scaler_data = MinMaxScaler().fit_transform(scaler_data)
    #return scaler_data, raw_data.target[index]
    return scaler_data, raw_data.target

def exec(nb, init=80, W=100):
    """
    :type init: int
    :type W: int
    """
    data, target = fetch('http', False)
    print("data-samples %s " % (data.shape[0]))
    for idx, ele in enumerate(target):
        if ele == b'normal.':
            target[idx] = 0
        else:
            target[idx] = 1
    if _Debug == True:
        pdb.set_trace()
    #record
    target = target.astype(int)
    predict = np.empty((data.shape[0],), dtype=float)

    model = loop(data[:init], W, extent=3, n_neighbors=nb).fit()
    print("model fit with init %s data samples" % (init))
    predict[:init] = model.local_outlier_probabilities
    ITER = len(data) - init
    for i in range(ITER):
        print(data[init+i])
        print("model init with %s th samples" % (init + i))
        val = model.insert(data[init+i])
        predict[init+i] = val
    fpr, tpr, thres = roc_curve(target, predict, pos_label=1)
    auc_score = auc(fpr, tpr)
    return auc_score

    
if __name__ == "__main__":
    init_ls = [80]
    W_ls = [100]
    nb_ls = [35]
    eps = open('http_experments.txt', 'a')
    for init, W in zip(init_ls, W_ls):
        for nb in nb_ls:
            score = exec(nb, init, W)
            eps.write("%s %s %s %s\n" % (init, W, nb, score))
    eps.close()

        

