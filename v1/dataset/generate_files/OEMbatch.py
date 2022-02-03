#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import distance_matrix


class OEM_batch:
    """
    On line Elastic Similarity class

    Parameters
    ----------
    R : 1-D array_like
        time series pattern
    w : float
        On-line similarity measure memory. Must be between 0 and 1.
    rateX : float, optional, default: 1
        Reference Time series generate_files rate
    rateY : float, optional, default: None
        Query Time series generate_files rate. If rateY = None it takes rateX value
    dist : string, optional, default: 'euclidean'
        cost function
        OPTIONS:
        'euclidean': np.sqrt((x_i - y_j)**2)
        'edr':       1 if abs(x_i - y_j) >= epsilon else 0
        'edit':      1 if x_i != y_j else 0
        'erp':       abs(x_i - y_j) if abs(x_i - y_j) >= epsilon else 0
    epsilon : float, optional, default: None
        edr threshold parameter
    """

    def __init__(self, R, w, dist='euclidean', epsilon=None):

        if dist in ['euclidean', 'edr', 'erp', 'edit', 'euclidean_new']:
            self.dist = dist
        else:
            raise ValueError('Not valid dist value')

        if isinstance(R, (np.ndarray)) and R.size > 2:
            self.R = R

        if (w >= 0) and (w <= 1):
            self.w = w
        else:
            raise ValueError('w must be between 0 and 1')

        if (epsilon is None) and (dist in ['edr', 'erp']):
            raise ValueError(
                'for dist edr or erp epsilon must be a non negative float')
        elif (epsilon is not None) and epsilon < 0:
            raise ValueError('epsilon must be a non negative float')
        self.epsilon = epsilon

    def init_dist(self, S):
        """
        Initial Similarity Measure

        Parameters
        ----------
        S : 1-D array_like
            Array containing time series observations

        Returns
        -------
        dist: float
            Elastic Similarity Measure between R (pattern time series) and S (query time series)
        """

        if isinstance(S, (np.ndarray)) and S.size > 2:
            pass
        else:
            raise ValueError('S time series must have more than 2 instances')

        # Compute point-wise distance
        if self.dist == 'euclidean':
            RS = self.__euclidean(self.R, S)
        if self.dist == 'euclidean_new':
            RS = self.__euclidean_new(self.R, S)
        elif self.dist == 'edr':
            RS = self.__edr(self.R, S, self.epsilon)
        elif self.dist == 'erp':
            RS = self.__erp(self.R, S, self.epsilon)
        elif self.dist == 'edit':
            RS = self.__edit(self.R, S)

        # compute recursive distance matrix using dynamic programing
        r, s = np.shape(RS)

        # Solve first row
        for j in range(1, s):
            RS[0, j] += self.w * RS[0, j - 1]

        # Solve first column
        for i in range(1, r):
            RS[i, 0] += RS[i - 1, 0]

        # Solve the rest
        for i in range(1, r):
            for j in range(1, s):
                RS[i, j] += np.min([RS[i - 1, j], self.w *
                                    RS[i - 1, j - 1], self.w * RS[i, j - 1]])

        # save statistics
        self.dtwR = RS[:, -1]

        return RS

    def update_dist(self, Y):
        '''
        Add new observations to query time series

        Parameters
        ----------
        Y : array_like or float
            new query time series observation(s)

        Returns
        -------
        dist : float
            Updated distance

        Warning: the time series have to be non-empty
        (at least composed by a single measure)


          -----------
        R | RS | RY |
          -----------
            S    Y
        '''

        if isinstance(Y, (np.ndarray)):
            pass
        else:
            Y = np.array([Y])

        # Solve RY
        dtwRY = self._solveRY(Y, self.dtwR)

        # Save statistics
        self.dtwR = dtwRY

        return dtwRY

    def __euclidean(self, X, Y):
        Y_tmp, X_tmp = np.meshgrid(Y, X)
        XY = np.sqrt((X_tmp - Y_tmp)**2)
        return XY

    def __euclidean_new(self, X, Y):
        return distance_matrix(X, Y, p=2)**2

    def __edr(self, X, Y, epsilon):
        Y_tmp, X_tmp = np.meshgrid(Y, X)
        XY = 1.0 * (abs(X_tmp - Y_tmp) < epsilon)
        return XY

    def __erp(self, X, Y, epsilon):
        Y_tmp, X_tmp = np.meshgrid(Y, X)
        XY = abs(X_tmp - Y_tmp)
        XY[XY < epsilon] = 0
        return XY

    def __edit(self, X, Y):
        Y_tmp, X_tmp = np.meshgrid(Y, X)
        XY = 1.0 * (X_tmp == Y_tmp)
        return XY

    def _solveRY(self, Y, dtwR):
        '''
        R, Y: to (partial) time series
         -----------
        R|prev| RY |
         -----------
           S    Y

        dtwR: partial solutions of DTW(R,S)
        iniI: Index of the first point of R in the complete time series
        iniJ: Index of the first point of Y in the complete time series

        * Warning *: R and Y have to be nonempty (partial) series
        '''

        # Compute point-wise distance
        if self.dist == 'euclidean':
            RY = self.__euclidean(self.R, Y)
        if self.dist == 'euclidean_new':
            RY = self.__euclidean_new(self.R, Y)
        elif self.dist == 'edr':
            RY = self.__edr(self.R, Y, self.epsilon)
        elif self.dist == 'erp':
            RY = self.__erp(self.R, Y, self.epsilon)
        elif self.dist == 'edit':
            RY = self.__edit(self.R, Y)

        r, n = np.shape(RY)

        # First first column
        RY[0, 0] += self.w * dtwR[0]
        for i in range(1, r):
            RY[i, 0] += np.min([
                RY[i - 1, 0], self.w * dtwR[i - 1], self.w * dtwR[i]])

        # Solve first row
        for j in range(1, n):
            RY[0, j] += self.w * RY[0, j - 1]

        # Solve the rest
        for j in range(1, n):
            for i in range(1, r):
                RY[i, j] += np.min([RY[i - 1, j], self.w *
                                    RY[i - 1, j - 1], self.w * RY[i, j - 1]])

        return RY[:, -1]
