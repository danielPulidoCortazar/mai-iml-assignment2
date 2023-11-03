import numpy as np
import sys


def sn_score(D, C, labels):
    xi = D.shape[1]
    summe = 1.0
    for i, data in enumerate(D):
        for i_xi in range(xi):
            summe = summe + np.absolute(data[i_xi] - C[int(labels[i])][i_xi])
    return int(summe * summe)


def kmeans_labels(D, k, max_iter, distance):
    len_D = len(D)  # number of Datapoints
    xi = D.shape[1]  # number of Dimension

    # create random points in the range of all D variables
    min = np.min(D, axis=0)
    max = np.max(D, axis=0)
    C = np.random.uniform(low=min, high=max, size=(k, xi))

    # create helper varibales to store the history of centroid points
    C_store = [C]
    C_first = C.copy()

    # iteration loop
    for n in range(max_iter):
        labels = np.zeros(len_D)

        # Loop through datapoints
        for i in range(len_D):
            # Second loop: compute distance for  Datapoint to every centroid
            dist = np.zeros(k)
            for ki, centroid in enumerate(C):
                dist[ki] = distance(D[i], np.array(centroid))
            labels[i] = np.argmin(dist)

        # calculate new centroids
        C_new = []
        for i in range(k):
            D_ofclass = D[(labels == i)]
            if D_ofclass.size != 0:
                new_center = np.sum(D_ofclass, axis=0) / len(D_ofclass)
                C_new.append(new_center)
            else:
                C_new.append(C[i])

        # store centroid history
        C_store.append(C_new)

        # break loop if centroids do not change anymore
        if np.array_equal(C_new, C):
            break

        # overwrite new centroid position
        C = C_new
    return labels, C, C_first, C_store

def get_mode(arr):
    cnt = {}
    for d in arr:
        key = str(d)
        if key in cnt:
            cnt[key][0] = cnt[key][0] + 1
        else:
            cnt[key] = [1, d]
    ma = [0]
    currentTie = []
    for key in cnt:
        if ma[0] < cnt[key][0]:
            ma = cnt[key]
            currentTie = [cnt[key]]
        elif ma[0] == cnt[key][0]:
            currentTie.append(cnt[key])
    if len(currentTie) == 1:
        return ma[1]
    vals = [e[1] for e in currentTie]
    vals.sort()
    return vals[round(len(vals)/2)]
    

def kmodes_labels(D, k, max_iter):
    len_D = len(D)  # number of Datapoints
    xi = D.shape[1]  # number of Dimension

    # create random points in the range of all D variables
    mi = np.min(D, axis=0)
    ma = np.max(D, axis=0)
    C = []
    for i in range(xi):
        C.append(np.random.randint(ma[i] - mi[i] + 1, size=(k)) + mi[i])
    C = np.array(C)
    C = np.transpose(C)

    # create helper varibales to store the history of centroid points
    C_store = [C]
    C_first = C.copy()

    # iteration loop
    for _ in range(max_iter):
        labels = np.zeros(len_D)

        # Loop through datapoints
        for i in range(len_D):
            # Second loop: compute distance for  Datapoint to every centroid
            distance = np.zeros(k)
            for ki, centroid in enumerate(C):
                ## Hamming Distance
                distance[ki] = np.sum(D[i] != np.array(centroid))
            labels[i] = np.argmin(distance)

        # calculate new centroids
        C_new = []
        for i in range(k):
            D_ofclass = D[(labels == i)]
            if D_ofclass.size != 0:
                transposed  = D_ofclass.T
                new_center = []
                for feature in transposed:
                    new_center.append(get_mode(feature))
                C_new.append(np.array(new_center))
            else:
                C_new.append(C[i])

        # store centroid history
        C_store.append(C_new)

        # break loop if centroids do not change anymore
        if np.array_equal(C_new, C):
            break

        # overwrite new centroid position
        C = C_new
    return labels, C, C_first, C_store


def manhattan(p1, p2):
    return np.abs(p1 - np.array(p2)).sum(-1)


def cosine(p1, p2):
    return 1.0 - np.dot(p2, p1) / (np.linalg.norm(p2) * np.linalg.norm(p1))


def euclidean(p1, p2):
    summe = 0.0
    for i_xi in range(len(p1)):
        summe = summe + abs((p2[i_xi] - p1[i_xi]) ** 2)
    return np.sqrt(summe)


def hamming(p1, p2):
    return np.sum(p1 != np.array(p2))


# FINAL WRAPPER FUNCTION
def kmeans(features, n_clusters=2, distance='manhattan', max_iter=100, n_runs=10):
    """takes the feature array and number of clusters; and returns an array of labels and a 3d array of centroids
    where the first dimension is a temporal dimension of how the centroids moved overtime"""

    if distance == 'cosine':
        distance = cosine
    elif distance == 'euclidean':
        distance = euclidean
    else:
        distance = manhattan

    D = np.array(features, dtype='float64')  # array of the input Data
    sn_list = []
    C_list = []
    labels_list = []
    C_first_list = []
    C_store_list = []
    for _ in range(n_runs):
        labels, C, C_first, C_store = kmeans_labels(D, n_clusters, max_iter, distance)
        sn = sn_score(D, C, labels)
        sn_list.append(sn)
        labels_list.append(labels)
        C_list.append(C)
        C_first_list.append(C_first)
        C_store_list.append(C_store)

    best_run = int(np.argmin(sn_list))
    labels_best = labels_list[best_run]
    C_store_best = C_store_list[best_run]

    return labels_best, C_store_best


# FINAL WRAPPER FUNCTION
def kmodes(features, n_clusters=2, max_iter=100):
    '''takes the feature array and number of clusters; and returns an array of labels and a 3d array of centroids where the first dimension is a temporal dimension of how the centroids moved overtime'''

    n = 10  # number of runs to incerase performance

    D = np.array(features)  # array of the input Data
    sn_list = []
    C_list = []
    labels_list = []
    C_first_list = []
    C_store_list = []
    for _ in range(n):
        labels, C, C_first, C_store = kmodes_labels(D, n_clusters, max_iter)
        sn = sn_score(D, C, labels)
        sn_list.append(sn)
        labels_list.append(labels)
        C_list.append(C)
        C_first_list.append(C_first)
        C_store_list.append(C_store)

    best_run = int(np.argmin(sn_list))
    labels_best = labels_list[best_run]
    C_store_best = C_store_list[best_run]

    return labels_best, C_store_best
