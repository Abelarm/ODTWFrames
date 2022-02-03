import numpy as np

from dataset.generate_files.OEMbatch import OEM_batch

def refMedoids(data):

    """
    Finds medoid time series in data.
    Arguments:
        dataPath {string} -- path of the data file
    """

    # load data
    labels = data[:, 0]
    data = data[:, 1:]
    numclass = np.unique(labels)

    medoidIDX = np.zeros((len(numclass),), dtype=int)
    aux = 0
    for label in numclass:
        # compute dtw among time series.
        tsIdx = np.where(labels == label)[0]
        dataSubset = data[tsIdx, :]
        distMAt = np.zeros((len(tsIdx), len(tsIdx)))
        for row in range(len(tsIdx)):
            for col in range(row + 1, len(tsIdx)):
                distMAt[row, col] = _dtw(
                    dataSubset[row, :], dataSubset[col, :])
        # find medoid
        medoid = np.argmin(np.sum(distMAt.T + distMAt, axis=0))
        medoidIDX[aux] = tsIdx[medoid]
        aux += 1
    return medoidIDX


def _dtw(x, y):
    """
    dtw between x and y time series

    Arguments:
        x, y {1-d array} -- time series

    Returns:
        float -- dtw between x and y
    """

    Y, X = np.meshgrid(y, x)

    RS = np.sqrt((X - Y) ** 2)
    r, s = np.shape(RS)

    # Solve first row
    for j in range(1, s):
        RS[0, j] += RS[0, j - 1]

    # Solve first column
    for i in range(1, r):
        RS[i, 0] += RS[i - 1, 0]

    # Solve the rest
    for i in range(1, r):
        for j in range(1, s):
            RS[i, j] += np.min([RS[i - 1, j], RS[i - 1, j - 1], RS[i, j - 1]])

    return RS[-1, -1]

def compute_odtw_distance_matrix(ref, stream, rho, dist="euclidean"):
    """
    Compute distance matrix.
    :param refmat: reference pattern
    :param stream: stream time series
    :param rho: memory
    :return: distance matrices
    """

    # print('Computing ODTW distance matrix')

    distMat = np.zeros((ref.shape[0], stream.shape[0]))
    odtw = OEM_batch(ref, rho, dist=dist)
    distMat[:, :3] = odtw.init_dist(stream[:3])

    for point in range(3, stream.shape[0]):
        if type(stream[point]) is list or type(stream[point]) is np.ndarray:
            tmp_stream = np.expand_dims(stream[point], axis=0)
        else:
            tmp_stream = [stream[point]]
        distMat[:, point] = odtw.update_dist(tmp_stream)

    return distMat
