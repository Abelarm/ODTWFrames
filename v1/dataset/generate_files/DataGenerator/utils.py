import matplotlib.pyplot as plt
import numpy as np


def compute_classpercentages(name):
    if name == 'arma':
        classpercentages = [1.0 / 8.0] * 8
        return np.array(classpercentages)
    elif name == 'synthetic_control':
        classpercentages = [1.0 / 6.0] * 6
        return np.array(classpercentages)
    elif (name == 'sines') or (name == 'kohlerlorenz'):
        classpercentages = [1.0 / 5.0] * 5
        return np.array(classpercentages)
    elif name == 'cbf':
        classpercentages = [1.0 / 3.0] * 3
        return np.array(classpercentages)
    elif (name == 'two_patterns') or (name == 'rational') or (name == 'seasonal'):
        classpercentages = [1.0 / 4.0] * 4
        return np.array(classpercentages)
    elif name == 'two_patterns2':
        classpercentages = [1.0 / 2.0] * 2
        return np.array(classpercentages)
    elif name == 'gunpoint':
        classpercentages = [1.0 / 2.0] * 2
        return np.array(classpercentages)


def identifyCutPoints(length, cycles):
    """
    Generates a boolean array that identifies points where one cycle finishes or starts
    length :: number of points in a time series
    cycles :: number of cycles in the the stream
    return :: cutPoints
    """

    cutPoints = np.zeros((length * cycles,))
    cutPoints[length * np.arange(1, cycles)] = 1
    cutPoints[length * np.arange(1, cycles + 1) - 1] = 1

    return cutPoints


def extractFrames(cutPoints, distMat, window_size):
    """
    Extracts frames from distMat, according to cutPoints
    cutPoints :: array of booleans. Determines end/start of a cycle
    distMat :: matrix from which frames are  extracted.
    window_size :: frames wide.
    """

    frames0, frames1 = [], []
    tp0, tp1 = [], []
    for t in range(window_size, cutPoints.size + 1):
        if np.any(cutPoints[t - window_size:t] == 1):
            frames1.append(distMat[:, t - window_size:t])
            tp1.append(t)
        else:
            if not np.any(cutPoints[t:t + window_size] == 1):
                frames0.append(distMat[:, t - window_size:t])
                tp0.append(t)

    return frames0, tp0, frames1, tp1


def splitMat_2_streamingFrames(mat, length, num_cycles, window_size, normalization=None):
    """
    Splits matrix into a set of continuos frames along axis 1, and assigns label 0 (no cut)
    or 1 (cut) to generated frames.
    ! use this function for the validation/test of the DNN

    Arguments:
        mat {2-d array} -- matrix to be splitted
        length {integer} -- length of each pattern
        num_cycles {integer} -- number of cycles in the stream
        window_size {interger} -- wide of the frame
        normalization {optional. string} -- apply normalization to generated frames. Default None
        other options: 'norm' (minmax normalization) and 'stand' (standard normalization). 
    """

    cutPoints = identifyCutPoints(length, num_cycles)
    frames = np.zeros((mat.shape[1] - window_size, mat.shape[0], window_size, mat.shape[-1]))
    labels = np.zeros((mat.shape[1] - window_size,))

    for t in range(window_size, mat.shape[1]):
        frames[t - window_size, :, :, :] = mat[:, t - window_size:t, :]
        if np.any(cutPoints[(t - window_size):min([t + window_size, mat.shape[1]])] == 1):
            labels[t - window_size] = 1

    return frames, labels


def matNorm(mat, axis=None):
    if axis is None:
        mat = (mat - np.min(mat)) / (np.max(mat) - np.min(mat))
        return mat

    if axis == 0:
        minVec = np.min(mat, axis=0)
        maxVec = np.max(mat, axis=0)
        for c in range(minVec.size):
            mat[:, c] = (mat[:, c] - minVec[c]) / (maxVec[c] - minVec[c])
        return mat

    if axis == 1:
        minVec = np.min(mat, axis=1)
        maxVec = np.max(mat, axis=1)
        for l in range(minVec.size):
            mat[l, :] = (mat[l, :] - minVec[l]) / (maxVec[l] - minVec[l])
        return mat


def matStand(mat, axis=None):
    if axis is None:
        mat = (mat - np.mean(mat)) / np.std(mat)
        return mat

    if axis == 0:
        meanVec = np.mean(mat, axis=0)
        stdVec = np.std(mat, axis=0)
        for c in range(meanVec.size):
            mat[:, c] = (mat[:, c] - meanVec[c]) / stdVec[c]
        return mat

    if axis == 1:
        meanVec = np.mean(mat, axis=1)
        stdVec = np.std(mat, axis=1)
        for l in range(meanVec.size):
            mat[l, :] = (mat[l, :] - meanVec[l]) / stdVec[l]
        return mat


def subsetFrames(allFrames, idFrames, num, seed):
    np.random.seed(seed)
    aux = np.random.choice(range(len(allFrames)), size=num, replace=False)
    subsetF, subsetID = [], []
    for i in aux:
        subsetF.append(allFrames[i])
        subsetID.append(idFrames[i])
    return subsetF, subsetID
