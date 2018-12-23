import os
import sys

import numpy as np
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_curve, auc
from PyNomaly.loop import LocalOutlierProbability as loop
import pdb

_Debug = False

def fetch(dataset='smtp', fetch_percent_10=True):
    if _Debug == True:
        pdb.set_trace()
    raw_data = fetch_kddcup99(subset=dataset, percent10=fetch_percent_10)
    unique_data, index = np.unique(raw_data.data.astype(float), axis=0, return_index=True)
    scaler_data = StandardScaler().fit_transform(raw_data.data[index])
    scaler_data = MinMaxScaler().fit_transform(scaler_data)
    scaler_data = raw_data.data[index].astype(float)
    return scaler_data, raw_data.target[index]

def exec():
    """
    :type init: int
    """
    data, target = fetch('smtp', False)
    for idx, ele in enumerate(target):
        if ele == b'normal.':
            target[idx] = 0
        else:
            target[idx] = 1
    if _Debug == True:
        pdb.set_trace()
    #record
    target = target.astype(int)
    model = loop(data, extent=3, n_neighbors=4).fit()
    # 8 0.9034
    # 7 0.9175
    # 6 0.9240
    # 5 0.9314
    # 4 0.9097
    predict = model.local_outlier_probabilities
# #   with open("target", 'w') as file:
#        for line in target:
#            file.write(str(line)+'\n')
#
#    model = loop(data[:init], 400, extent=3, n_neighbors=8).fit()
#    print("model fit with init %s data samples" % (init))
#    f = open("score", 'w')
#    for ele in model.local_outlier_probabilities:
#        f.write(str(ele) + '\n')
#    predict[:init] = model.local_outlier_probabilities
#    ITER = len(data) - init
#    for i in range(ITER):
#        print(data[init+i])
#        print("model init with %s th samples" % (init + i))
#        val = model.insert(data[init+i])
#        predict[init+i] = val
#        f.write(str(val) + '\n')
#    f.close()
    fpr, tpr, thres = roc_curve(target, predict, pos_label=1)
    auc_score = auc(fpr, tpr)
    print(auc_score)
#
    
if __name__ == "__main__":
    exec()
    

        

