import itertools
from math import erf, exp
import numpy as np
import sys
import pdb

_Debug = False
class LocalOutlierProbability(object):
    """
    :param data: a Pandas DataFrame or Numpy array of float data
    :param extent: an integer value [1, 2, 3] that controls the statistical extent, e.g. lambda times the standard deviation from the mean (optional, default 3)
    :param n_neighbors: the total number of neighbors to consider w.r.t. each sample (optional, default 10)
    :param cluster_labels: a numpy array of cluster assignments w.r.t. each sample (optional, default None)
    :return:
   """

    class check:

        @staticmethod
        def data(obj):
            if obj.__class__.__name__ == 'DataFrame':
                points_vector = obj.values
                return points_vector
            elif obj.__class__.__name__ == 'ndarray':
                points_vector = obj
                return points_vector
            else:
                warnings.warn(
                    'Provided data must be in ndarray or DataFrame.',
                    UserWarning)
                if isinstance(obj, list):
                    points_vector = np.array(obj)
                    return points_vector
                points_vector = np.array([obj])
                return points_vector

        @staticmethod
        def cluster_size(obj):
            c_labels = obj._cluster_labels()
            for cluster_id in set(c_labels):
                c_size = np.where(c_labels == cluster_id)[0].shape[0]
                if c_size <= obj.n_neighbors:
                    warnings.warn(
                        'Number of neighbors specified larger than smallest cluster. Specify a number of neighbors smaller than the smallest cluster size (observations in smallest cluster minus one).',
                        UserWarning)
                    return False

        @staticmethod
        def n_neighbors(obj, set_neighbors=False):
            if not obj.n_neighbors > 0:
                warnings.warn('n_neighbors must be greater than 0.',
                              UserWarning)
                return False
            elif obj.n_neighbors >= obj._n_observations():
                if set_neighbors is True:
                    obj.n_neighbors = obj._n_observations() - 1
                warnings.warn(
                    'n_neighbors must be less than the number of observations. Fit with ' + str(
                        obj.n_neighbors) + ' instead.', UserWarning)
                return True

        @staticmethod
        def extent(obj):
            if obj.extent not in [1, 2, 3]:
                warnings.warn(
                    'extent parameter (lambda) must be 1, 2, or 3.',
                    UserWarning)
                return False
            else:
                return True

        @staticmethod
        def missing_values(obj):
            if np.any(np.isnan(obj.data)):
                warnings.warn(
                    'Method does not support missing values in input data. ',
                    UserWarning)
                return False
            else:
                return True

        @staticmethod
        def fit(obj):
            if obj.points_vector is None:
                warnings.warn(
                    "Must fit on historical data by calling fit() prior to calling stream(x).",
                    UserWarning)
                return False
            else:
                return True

        @staticmethod
        def no_cluster_labels(obj):
            if len(set(obj._cluster_labels())) > 1:
                warnings.warn(
                    'Stream approach does not support clustered data. Automatically refit using single cluster of points.',
                    UserWarning)
                return False
            else:
                return True

    def accepts(*types):
        def decorator(f):
            assert len(types) == f.__code__.co_argcount

            def new_f(*args, **kwds):
                for (a, t) in zip(args, types):
                    if type(a).__name__ == 'DataFrame':
                        a = np.array(a)
                    if isinstance(a, t) is False:
                        warnings.warn("Argument %r is not of type %s" % (a, t),
                                      UserWarning)
                opt_types = {
                    'W':{
                        'type': types[2]
                    },
                    'extent': {
                        'type': types[3]
                    },
                    'n_neighbors': {
                        'type': types[4]
                    },
                    'cluster_labels': {
                        'type': types[5]
                    }
                }
                for x in kwds:
                    opt_types[x]['value'] = kwds[x]
                for k in opt_types:
                    try:
                        if isinstance(opt_types[k]['value'],
                                      opt_types[k]['type']) is False:
                            warnings.warn("Argument %r is not of type %s" % (
                                k, opt_types[k]['type']), UserWarning)
                    except KeyError:
                        pass
                return f(*args, **kwds)

            new_f.__name__ = f.__name__
            return new_f

        return decorator

    @accepts(object, np.ndarray, (int, np.integer), (int, np.integer), (int, np.integer), list)
    def __init__(self, data, W, extent=3, n_neighbors=10, cluster_labels=None):
        self.data = data
        self.W = W
        self.extent = extent
        self.n_neighbors = n_neighbors
        self.cluster_labels = cluster_labels
        self.points_vector = None
        self.prob_distances = None
        self.prob_distances_ev = None
        self.norm_prob_local_outlier_factor = None
        self.local_outlier_probabilities = None
        self._objects = {}

        self.check.data(self.data)
        self.check.n_neighbors(self)
        self.check.cluster_size(self)
        self.check.extent(self)
        self.check.missing_values(self)

    @staticmethod
    def _standard_distance(cardinality, sum_squared_distance):
        st_dist = np.array(
            [np.sqrt(i) for i in np.divide(sum_squared_distance, cardinality)])
        return st_dist

    @staticmethod
    def _prob_distance(extent, standard_distance):
        return extent * standard_distance

    @staticmethod
    def _prob_outlier_factor(probabilistic_distance, ev_prob_dist):
        if np.all(probabilistic_distance == ev_prob_dist):
            return np.zeros(probabilistic_distance.shape)
        else:
            try:
                return (probabilistic_distance / ev_prob_dist) - 1.
            except ZeroDivisionError:
                print(probabilistic_distance)
                print(ev_prob_dist)
                exit(1)
                
    @staticmethod
    def _norm_prob_outlier_factor(extent, ev_probabilistic_outlier_factor):
        ev_probabilistic_outlier_factor = [i for i in
                                           ev_probabilistic_outlier_factor]
        return extent * np.sqrt(ev_probabilistic_outlier_factor)

    @staticmethod
    def _local_outlier_probability(plof_val, nplof_val):
        erf_vec = np.vectorize(erf)
        if np.all(plof_val == nplof_val):
            return np.zeros(plof_val.shape)
        else:
            return np.maximum(0, erf_vec(plof_val / (nplof_val * np.sqrt(2.))))

    def _n_observations(self):
        return len(self.data)

    def _store(self):
        return np.empty([self._n_observations(), 3], dtype=object)

    def _cluster_labels(self):
        if self.cluster_labels is None:
            return np.array([0] * len(self.data))
        else:
            return np.array(self.cluster_labels)

    @staticmethod
    def _euclidean(vector1, vector2):
        diff = vector1 - vector2
        return np.dot(diff, diff) ** 0.5

    def _distances(self, data_store):
        distances = np.full([self._n_observations(), self.n_neighbors], 9e10,
                            dtype=float)
        indexes = np.full([self._n_observations(), self.n_neighbors], 9e10,
                          dtype=float)
        self.points_vector = self.check.data(self.data)
        for cluster_id in set(self._cluster_labels()):
            indices = np.where(self._cluster_labels() == cluster_id)
            clust_points_vector = self.points_vector.take(indices, axis=0)[0]
            pairs = itertools.permutations(
                np.ndindex(clust_points_vector.shape[0]), 2)
            for p in pairs:
                d = self._euclidean(clust_points_vector[p[0]],
                                    clust_points_vector[p[1]])
                idx = indices[0][p[0]]
                idx_max = np.argmax(distances[idx])
                if d < distances[idx][idx_max]:
                    distances[idx][idx_max] = d
                    indexes[idx][idx_max] = p[1][0]
        for vec, cluster_id in zip(range(distances.shape[0]),
                                   self._cluster_labels()):
            data_store[vec][0] = cluster_id
            data_store[vec][1] = distances[vec]
            data_store[vec][2] = indexes[vec]
        return data_store

    def _ssd(self, data_store):
        self.cluster_labels_u = np.unique(data_store[:, 0])
        ssd_array = np.empty([self._n_observations(), 1])
        for cluster_id in self.cluster_labels_u:
            indices = np.where(data_store[:, 0] == cluster_id)
            cluster_distances = np.take(data_store[:, 1], indices).tolist()
            ssd = np.sum(np.power(cluster_distances[0], 2), axis=1)
            for i, j in zip(indices[0], ssd):
                ssd_array[i] = j
        data_store = np.hstack((data_store, ssd_array))
        return data_store

    def _standard_distances(self, data_store):
        cardinality = np.array([self.n_neighbors] * self._n_observations())
        return np.hstack(
            (data_store,
             np.array([np.apply_along_axis(self._standard_distance, 0,
                                           cardinality, data_store[:, 3])]).T))

    def _prob_distances(self, data_store):
        return np.hstack((data_store, np.array(
            [self._prob_distance(self.extent, data_store[:, 4])]).T))

    def _prob_distances_ev(self, data_store):
        prob_set_distance_ev = np.empty([self._n_observations(), 1])
        for cluster_id in self.cluster_labels_u:
            indices = np.where(data_store[:, 0] == cluster_id)[0]
            for index in indices:
                nbrhood = data_store[index][2].astype(int)
                nbrhood_prob_distances = np.take(data_store[:, 5], nbrhood).astype(float)
                nbrhood_prob_distances_nonan = nbrhood_prob_distances[np.logical_not(np.isnan(nbrhood_prob_distances))]
                prob_set_distance_ev[index] = np.mean(nbrhood_prob_distances_nonan)
        self.prob_distances_ev = prob_set_distance_ev
        data_store = np.hstack((data_store, prob_set_distance_ev))
        return data_store

    def _prob_local_outlier_factors(self, data_store):
        return np.hstack(
            (data_store,
             np.array([np.apply_along_axis(self._prob_outlier_factor, 0,
                                           data_store[:, 5],
                                           data_store[:, 6])]).T))

    def _prob_local_outlier_factors_ev(self, data_store):
        prob_local_outlier_factor_ev_dict = {}
        for cluster_id in self.cluster_labels_u:
            indices = np.where(data_store[:, 0] == cluster_id)
            prob_local_outlier_factors = np.take(data_store[:, 7],
                                                 indices).astype(float)
            prob_local_outlier_factors_nonan = prob_local_outlier_factors[
                np.logical_not(np.isnan(prob_local_outlier_factors))]
            prob_local_outlier_factor_ev_dict[cluster_id] = np.sum(
                np.power(prob_local_outlier_factors_nonan, 2)) / \
                                                            float(
                                                                prob_local_outlier_factors_nonan.size)
        data_store = np.hstack(
            (data_store, np.array([[prob_local_outlier_factor_ev_dict[x] for x
                                    in data_store[:, 0].tolist()]]).T))
        return data_store

    def _norm_prob_local_outlier_factors(self, data_store):
        return np.hstack((data_store, np.array([self._norm_prob_outlier_factor(
            self.extent, data_store[:, 8])]).T))

    def _local_outlier_probabilities(self, data_store):
        return np.hstack(
            (data_store,
             np.array([np.apply_along_axis(self._local_outlier_probability, 0,
                                           data_store[:, 7],
                                           data_store[:, 9])]).T))

    def fit(self):

        self.check.data(self.data)
        self.check.n_neighbors(self, set_neighbors=True)
        self.check.cluster_size(self)
        if self.check.missing_values(self) is False:
            sys.exit()

        store = self._store()
        store = self._distances(store)
        store = self._ssd(store)
        store = self._standard_distances(store)
        store = self._prob_distances(store)
        self.data_store = store

        self.prob_distances = store[:, 5]
        store = self._prob_distances_ev(store)
        store = self._prob_local_outlier_factors(store)
        store = self._prob_local_outlier_factors_ev(store)
        store = self._norm_prob_local_outlier_factors(store)
        self.norm_prob_local_outlier_factor = np.max(store[:, 9])
        store = self._local_outlier_probabilities(store)
        self.local_outlier_probabilities = store[:, 10]

        return self

    def stream(self, x):

        if self.check.no_cluster_labels(self) is False:
            self.cluster_labels = np.array([0] * len(self.data))
            self.fit()

        if self.check.fit(self) is False:
            sys.exit()

        distances = np.full([1, self.n_neighbors], 9e10, dtype=float)
        point_vector = self.check.data(x)
        for p in range(0, self.points_vector.shape[0]):
            d = self._euclidean(self.points_vector[p, :], point_vector)
            idx_max = np.argmax(distances[0])
            if d < distances[0][idx_max]:
                distances[0][idx_max] = d
        ssd = np.sum(np.power(distances, 2))
        std_dist = np.sqrt(np.divide(ssd, self.n_neighbors))
        prob_dist = self._prob_distance(self.extent, std_dist)
        plof = self._prob_outlier_factor(prob_dist,
                                          np.mean(self.prob_distances_ev))
        loop = self._local_outlier_probability(
            plof, self.norm_prob_local_outlier_factor)

        return loop

    def insert(self, x):

        if self.check.fit(self) is False:
            sys.exit()

        if self._n_observations() == self.W:
            print("begin summarize..")
            self.summarize()
        self.data = np.vstack((self.data, x))

        point_vector = self.check.data(x)
        cluster_id = 0
        distances = np.full([1, self.n_neighbors], 9e10, dtype=float)
        indexes = np.full([1, self.n_neighbors], 9e10, dtype=float)
        distances_all = self.data_store[:, 1] #shallow copy
        indexes_all = self.data_store[:, 2] #shallow copy
        seen = []
        for p in range(0, self.points_vector.shape[0]):
            d = self._euclidean(self.points_vector[p, :], point_vector)
            idx_max = np.argmax(distances[0])
            if d<distances[0][idx_max]:
                distances[0][idx_max] = d
                indexes[0][idx_max] = p
            neighbor_idx_max = np.argmax(distances_all[p])
            if d < distances_all[p][neighbor_idx_max]:
                distances_all[p][neighbor_idx_max] = d
                indexes_all[p][neighbor_idx_max] = self.points_vector.shape[0]
                seen.append(p)
                self.data_store[p][3] = np.sum(np.power(self.data_store[p][1], 2))
                self.data_store[p][4] = np.sqrt(np.divide(self.data_store[p][3], self.n_neighbors))
                self.data_store[p][5] = self._prob_distance(self.extent, self.data_store[p][4])
        new_store = np.empty([1, 3], dtype=object)
        new_store[0][0] = cluster_id
        new_store[0][1] = distances[0]
        new_store[0][2] = indexes[0]
        self.points_vector = np.vstack((self.points_vector, point_vector))

        ssd = np.sum(np.power(distances, 2), axis=1).reshape(1,-1) #shape error
        new_store = np.hstack((new_store, ssd)) # 3
        std_dist = np.sqrt(np.divide(ssd, self.n_neighbors))
        new_store = np.hstack((new_store, std_dist)) # 4
        prob_dist = self._prob_distance(self.extent, std_dist)
        new_store = np.hstack((new_store, prob_dist)) # 5
        
        self.data_store = np.vstack((self.data_store, new_store))
        self.prob_distances = self.data_store[:, 5]
        data_store = self._prob_distances_ev(self.data_store)
        data_store = self._prob_local_outlier_factors(data_store)
        data_store = self._prob_local_outlier_factors_ev(data_store)

        data_store = self._norm_prob_local_outlier_factors(data_store)
        self.norm_prob_local_outlier_factor = np.max(data_store[:, 9])
        data_store = self._local_outlier_probabilities(data_store)
        self.local_outlier_probabilities = data_store[:, 10]
        if _Debug == True:
            pdb.set_trace()
        return self.local_outlier_probabilities[-1]

    def summarize(self):
        """
        :W integer
        """
        self.selectOptimalInstances()
        # update self.data_store
        self.check.data(self.data)
        self.check.n_neighbors(self, set_neighbors=True)
        self.check.cluster_size(self)
        store = self._store()
        store = self._distances(store)
        if _Debug == True:
            pdb.set_trace()
        store = self._ssd(store)
        store = self._standard_distances(store)
        store = self._prob_distances(store)
        self.data_store = store # samples * 6  

    @staticmethod
    def sigmoid(x):
        """
        :type x: np.ndarray
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def project(y, tempY):
        """
        :type y: ndarray
        :type tempY: ndarray
        """
        H_W = len(tempY)
        Q_W = H_W // 2
        pair = list(zip(tempY, np.arange(H_W)))
        pair = sorted(pair, key=lambda it:it[0], reverse=True)
        for i in range(Q_W):
            y[pair[i][1]] = True

    def selectOptimalInstances(self, MAX_ITERATION=30, LEARNING_RATE=0.3):
        """
        :type self.W threshold
        """
        H_W = self.W // 2
        Q_W = self.W // 4 # //2 before, fixed by lss in 2018.12.26
        #avgSumNNLOOP = 0
        #sumLOOP = 0
        y = np.array([False] * H_W)
        tempY = np.array([0.5] * H_W)
        sumNNLOOP = np.array([0.0] * H_W)  # float 
        self.local_outlier_probabilities = self.local_outlier_probabilities.astype(float)
        expSigLOOP = np.exp(self.sigmoid(self.local_outlier_probabilities[:H_W]))
        dis = self.data_store[:, 1].flatten()
        ind = self.data_store[:, 2].flatten()
        matrix = []
        for i in range(len(self.data_store)):
            tmp = list(zip(self.data_store[i][1], self.data_store[i][2]))
            tmp.sort(key=lambda it:it[0])
            matrix.append(tmp)
        matrix = np.array(matrix) #2*2*2
        distMatrix = matrix[:, :, 0]
        knnMatrix = matrix[:, :, 1].astype(int)

        for i in range(H_W):
            for k in range(self.n_neighbors):
                if knnMatrix[i][k] < H_W:
                    sumNNLOOP[i] += expSigLOOP[knnMatrix[i][k]]
        avgSumNNLOOP = np.average(sumNNLOOP)
        sumLOOP = np.sum(sumNNLOOP)

        def getApproximatedKthNNDist(INDEX, SUM_NN_LOOP, SUM_LOOP):
            """
            :type INDEX: const int
            :type SUM_NN_LOOP: const float
            :type SUM_LOOP: const float
            """
            #kthNNDistance = distMatrix[INDEX][knnMatrix[INDEX][self.n_neighbors-1]]
            kthNNDistance = distMatrix[INDEX][self.n_neighbors-1]
            #maxDistance = np.max(distMatrix[INDEX, :H_W])
            res = (kthNNDistance + (SUM_NN_LOOP / SUM_LOOP)*(maxDistance - kthNNDistance)) 
            return res / kthNNDistance

        approxRho = np.array([0.0] * H_W)
        gradients = np.array([0.0] * H_W)
        kthNNs = np.empty((H_W, ), dtype=int)
        for i in range(H_W):

            dist = np.empty((H_W, ), dtype=float)
            for idx, point in enumerate(self.points_vector[:H_W]):
                dist[idx] = self._euclidean(self.points_vector[i], point)
            maxDistance = np.max(dist)

            approxRho[i] = getApproximatedKthNNDist(i, sumNNLOOP[i], sumLOOP)
            kthNN = knnMatrix[i][self.n_neighbors - 1]
            kthNNs[i] = kthNN

            if sumNNLOOP[i] > avgSumNNLOOP:
                for j in range(H_W):
                    mid = 2*self.sigmoid(self.local_outlier_probabilities[i] * distMatrix[i][self.n_neighbors - 1])
                    if dist[j] > distMatrix[i][self.n_neighbors - 1] and dist[j] < mid:
                        kthNNs[i] = j
                        break

        for it in range(MAX_ITERATION):
            LEARNING_RATE *= 0.95
            sumY = np.sum(tempY)
            for i in range(H_W):
                sumDist = 0
                for n in range(H_W):
                    dist_n_i = self._euclidean(self.points_vector[n], self.points_vector[i])
                    if kthNNs[n] == i:
                        sumDist += tempY[n] * (
                                dist_n_i / distMatrix[n][self.n_neighbors - 1])
                gradients[i] = (sumDist + approxRho[i]) - exp(self.local_outlier_probabilities[i])\
                        + 0.001 * (sumY - Q_W)

                if tempY[i] > 1:
                    gradients[i] += 2 * (tempY[i] - 1)
                elif tempY[i] < 0:
                    gradients[i] += 2 * tempY[i]

            for i in range(H_W):
                tempY[i] -= LEARNING_RATE * gradients[i]
        self.project(y, tempY)
        if _Debug == True:
            pdb.set_trace()
        #remove Instance from dataset
        self.data = np.vstack((self.data[:H_W][y], self.data[H_W:]))





