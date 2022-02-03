import numpy as np

def coef_of_determination(ytrue, ypred):

    SS_res = np.sum((ypred - np.mean(ytrue))**2)
    SS_tot = np.sum((ytrue - np.mean(ytrue))**2)
    r2 = 1 - SS_res / SS_tot

    return r2