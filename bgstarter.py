from itertools import cycle, islice
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
# import seaborn
import pandas as pd
import _pickle as pickle
import numpy as np
import time

from spectral import utils, affinity, clustering

from sklearn.cluster import SpectralClustering

if __name__ == "__main__":
    start = time.time()

    methods = [
        (affinity.compute_affinity, 'Basic Affinity'),
        (affinity.com_aff_local_scaling, 'Affinity Local Scaling'),
        (affinity.automatic_prunning, 'Auto-pruning + LS'),
        # (partial(affinity.automatic_prunning, affinity=affinity.compute_affinity), 'Auto-pruning'),
    ]

    X_pd = pickle.load(open('/Users/wang/data/OpenAPS/cluster_train_index.pkl', 'rb'))
    X_pd_1000 = X_pd[:1000]
    X = X_pd_1000['sgvs'].values
    X = [x.tolist() for x in X]
    # print(X)
    # print('X.shape', X.shape)

    # clustering = SpectralClustering(n_clusters=2,assign_labels="discretize", random_state=0).fit(X)
    # print(clustering.labels_)
    
    X = np.array(X)
    N = X.shape[0]
    K = 3
    affinity, name = methods[2]
    A = affinity(X)
    print('A.shape', A.shape)
    print("SC  with %d clases with method %r" % (K, name))
    labels = clustering.spectral_clustering(A, K)
    X_pd_1000['label'] = labels

    pickle.dump(X_pd_1000, open('/Users/wang/data/OpenAPS/cluster_1000_3.pkl', 'wb'))

    print(labels)
    end = time.time()
    print('time last', end-start)