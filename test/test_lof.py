from sklearn.metrics import roc_curve, auc
from sklearn.datasets import fetch_kddcup99
from sklearn.neighbors import LocalOutlierFactor as lof
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

def exec(nb=20):
    data = np.loadtxt('../http.txt', delimiter=",")
    smtp = data[:, :3].astype(float)
    target = data[:, 4].astype(int)
    model = lof(n_neighbors=nb)
    model.fit_predict(smtp)
    score = -model.negative_outlier_factor_
    fpr, tpr, thres = roc_curve(target, score, pos_label=1)
    print(auc(fpr, tpr))

    scaler = StandardScaler().fit_transform(smtp)
    scaler = MinMaxScaler().fit_transform(scaler)
    #model = lof(n_neighbors=nb, metric='euclidean')
    model = lof(n_neighbors=nb)
    model.fit_predict(scaler)
    score = -model.negative_outlier_factor_
    fpr, tpr, thres = roc_curve(target, score, pos_label=1)
    print(auc(fpr, tpr))

if __name__ == "__main__":
    import sys
    nb = 20
    if len(sys.argv) > 1:
        nb = int(sys.argv[1])
    exec(nb)
