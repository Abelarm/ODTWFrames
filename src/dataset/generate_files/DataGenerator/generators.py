# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ---------------------------------- TIME SERIES GENERATOR -------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------

import numpy as np
from patsy.highlevel import dmatrix
from statsmodels.genmod.generalized_linear_model import GLM

from dataset.generate_files.DataGenerator.arima import sim_arima


def arma(length_series, max_shift, shift, cluster, seed=123):

    # The seed is selected.
    np.random.seed(seed)

    # The innovations are created. They are specific to this shape.
    innovations = np.random.randn(length_series + 100 + 2 * abs(max_shift))

    # An arma series is generated based on the innovations.
    serie = sim_arima(
        model=[(3, 0, 2), (1, -0.24, 0.1), (1, 1.2)],
        n=length_series + 100 + 2 * abs(max_shift),
        nstart=100,
        errors=innovations)

    # The series is shifted.
    serie = serie[(100 + abs(max_shift) + shift)
                   :(serie.size - abs(max_shift) + shift)]
    serie = serie.reset_index(drop=True)
    return serie

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------


def synthetic_control(length_series, shift, cluster, seed=None):

    np.random.seed(seed)

    def aux_fun(vector, b):
        vector[:b] = 0
        vector[b:] = 1
        return vector

    if cluster == 1:
        return (80 + 15 * np.sin((2 * np.pi) / (0.3 * length_series) * (shift + np.arange(1, length_series + 1))))

    if cluster == 2:
        return (80 + 0.4 * np.arange(1, length_series + 1) + shift)

    if cluster == 3:
        return (80 - 0.4 * np.arange(1, length_series + 1) + shift)

    if cluster == 4:
        cut_point = int(np.floor(length_series / 2.0) +
                        shift)  # (modified) cut point
        shape = aux_fun(np.arange(length_series), cut_point)  # basic shape
        return (80 + 10 * shape)

    if cluster == 5:
        cut_point = int(np.floor(length_series / 2.0) +
                        shift)  # (modified) cut point
        shape = aux_fun(np.arange(length_series), cut_point)  # basic shape
        return (90 - 10 * shape)

    if cluster == 6:
        return (80 + np.random.normal(loc=0, scale=3, size=length_series))

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------


def sines(length_series, shift, noise_level, cluster, seed=None):

    np.random.seed(seed)

    base = np.arange(1, length_series + 1)
    # A basic period for the sinusoidal form is generated
    T = 0.5 * length_series
    core = 2 * np.pi * (base + shift) / T
    series = np.sin(core)

    if cluster == 1:
        # series is multiplied by a rndom number
        series *= np.random.uniform(low=1, high=1.1, size=1)
        # noise is added
        if noise_level != 0:
            noise = np.random.normal(
                loc=0, scale=noise_level / 100, size=length_series)
            series += noise
        return series

    if cluster == 2:
        import statsmodels.api as sm
        lowess = sm.nonparametric.lowess
        series = lowess(series, base, frac=0.25)[:, 1]
        # series is multiplied by a random number
        series *= np.random.uniform(low=1, high=1.1, size=1)
        # noise is added
        if noise_level != 0:
            noise = np.random.normal(
                loc=0, scale=noise_level / 100, size=length_series)
            series += noise
        return series

    if cluster == 3:
        # series is multiplied by a rndom number
        series *= np.random.uniform(low=1, high=1.1, size=1)
        # noise is added
        if noise_level != 0:
            noise = np.random.normal(
                loc=0, scale=noise_level / 100, size=length_series)
            series += noise
        # The values  are above a  threshold are truncated
        threshold = np.random.uniform(low=0.9, high=0.99, size=1)
        series[series > threshold] = threshold
        return series

    if cluster == 4:
        # series is multiplied by a rndom number
        series *= np.random.uniform(low=1, high=1.1, size=1)
        # noise is added
        if noise_level != 0:
            noise = np.random.normal(
                loc=0, scale=noise_level / 100, size=length_series)
            series += noise
        # The values  are above a  threshold are truncated
        threshold = np.random.uniform(low=-0.99, high=-0.90, size=1)
        series[series < threshold] = threshold
        return series

    if cluster == 5:
        # series is multiplied by a rndom number
        series *= np.random.uniform(low=1, high=1.1, size=1)
        # noise is added
        if noise_level != 0:
            noise = np.random.normal(
                loc=0, scale=noise_level / 100, size=length_series)
            series += noise
        # A random number is added to a small part of the series
        cut_point1 = int(
            np.max([np.round(0.4 * length_series) + shift, 1]) - 1)
        cut_point2 = int(
            np.min([np.round(0.5 * length_series) + shift, length_series]) - 1)
        series[cut_point1:cut_point2] += np.random.uniform(
            low=0.2, high=0.3, size=1)
        series *= np.random.uniform(low=1, high=1.1, size=1)
        return series

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------


def kohlerlorenz(length_series, max_shift, shift, warp, cluster, seed=None):

    np.random.seed(seed)
    if cluster == 1:
        # Sinusoidal shape.
        series = np.random.normal(loc=1, scale=0.1, size=1) * 13 * np.sin(
            (5 * np.arange(length_series) + shift) / (0.1 * length_series))
        return series

    if cluster == 2:
        # The cut point is defined.
        cut_point = int(np.round(length_series / 2.0) + shift)
        # The possible errors for the cut point are defined.
        if cut_point < 0:
            cut_point = 0
        if cut_point >= length_series:
            cut_point = length_series - 1
        # The series is created
        series = np.arange(length_series)
        series[:cut_point] = series[:cut_point]**2 / 100
        series[cut_point:] = series[cut_point:]**2 / 530
        return series

    if cluster == 3:
        # The cut points are defined
        cut_point1 = int(np.round(length_series / 3.0) + (shift + warp)
                         * np.random.normal(loc=1, scale=0.1, size=1))
        cut_point2 = int(np.round(2 * length_series / 3.0) +
                         shift * np.random.normal(loc=1, scale=0.1, size=1))
        # The possible errors for the cut point are defined.
        if cut_point1 < 0:
            cut_point1 = 0
        if cut_point2 >= length_series:
            cut_point2 = length_series - 1
        # The piecewise constant function is defined.
        series = np.zeros((length_series,))
        series[:cut_point1] = 5
        series[cut_point1:cut_point2] = 12
        series[cut_point2:] = 8
        return(series)

    if cluster == 4:
        # The cut points are defined
        cut_points = [int(np.round((i * length_series) / 8.0) + shift *
                          np.random.normal(loc=1, scale=0.1, size=1)) for i in range(1, 6)]
        if cut_points[0] < 0:
            cut_points[0] = 0
        if cut_points[4] >= length_series:
            cut_points[4] = length_series - 1
        # The possible errors for the cut point are identified
        series = np.zeros((length_series,))
        series[:cut_points[0]] = np.random.normal(loc=15, scale=0.1, size=1) * np.sin(
            5 * np.random.normal(loc=1, scale=0.1, size=1) * np.arange(cut_points[0]))
        series[cut_points[0]:cut_points[1]] = np.random.normal(loc=15, scale=0.1, size=1) * np.sin(
            7 * np.random.normal(loc=1, scale=0.1, size=1) * np.arange(cut_points[0], cut_points[1]))
        series[cut_points[1]:cut_points[2]] = 10
        series[cut_points[2]:cut_points[3]] = 2
        series[cut_points[3]:cut_points[4]] = (
            np.arange(cut_points[3], cut_points[4]) / (length_series / 2.0))**2 / 2.0
        series[cut_points[4]:] = (
            np.arange(cut_points[4], length_series) / (length_series / 2.0))**2 / 2.0
        return series

    if cluster == 5:
        # The seed is selected.
        np.random.seed(seed)
        # The innovations are created. They are specific to this shape.
        innovations = np.random.randn(
            int(length_series + 100 + 2 * abs(max_shift)))
        # An arma series is generated based on the innovations.
        serie = sim_arima(model=[(3, 0, 2), (1, -0.24, 0.1), (1, 1.2)], n=int(
            length_series + 100 + 2 * abs(max_shift)), nstart=100, errors=innovations)
        # The series is shifted.
        serie = serie[(100 + abs(max_shift) + shift)
                       :(serie.size - abs(max_shift) + shift)]
        serie = serie.reset_index(drop=True)
        return serie

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------


def cbf(length_series, shift, warp, cluster, seed=None):

    np.random.seed(None)

    # The initial cut points are situated
    cut_point1 = np.floor(float(length_series) / 3)
    cut_point2 = np.floor(float(2 * length_series) / 3)

    # The cut point are moved depending on th shift and warp:
    mod_cut_point1 = int(np.max([cut_point1 + shift + warp, 1]))
    mod_cut_point2 = int(np.min([cut_point2 + shift + warp, length_series]))

    # The series is defined
    series = np.zeros((length_series,))
    series[mod_cut_point1 - 1:mod_cut_point2] = 1
    if cluster == 'cylinder':
        series *= (6.0 + np.random.randn(1))
    if cluster == 'bell':
        series *= (6.0 + np.random.randn(1)) * (np.arange(length_series) -
                                                mod_cut_point1) / (mod_cut_point2 - mod_cut_point1)
    if cluster == 'funnel':
        series *= (6.0 + np.random.randn(1)) * (mod_cut_point2 -
                                                np.arange(length_series)) / (mod_cut_point2 - mod_cut_point1)

    return series

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------


def two_patterns(length_series, shift, warp, cluster, seed=None):

    np.random.seed(seed)

    # Upward shape
    def us(t, l):
        vector = 5 * np.ones((t,))
        vector[:int(np.ceil(l / 2.0))] *= -1
        return vector

    # Downward shape
    def ds(t, l):
        vector = 5 * np.ones((t,))
        vector[int(np.ceil(l / 2.0)):] *= -1
        return vector

    # Define cut points
    t1 = int(np.floor(length_series / 3.0) + shift)
    t2 = int(np.floor(2 * length_series / 3.0) + shift)
    l1 = np.round(0.1 * length_series + warp).astype(np.uint8)
    l2 = l1

    # The series is created (common part for all clusters)
    series = np.zeros((length_series,))
    # EXTRA NOISE.
    if t1 > 0:
        series[:t1] = np.random.randn(t1)
    if t2 > (t1 + l1):
        series[(t1 + l1):t2] = np.random.randn(t2 - t1 - l1)
    series[(t2 + l2):] = np.random.randn(length_series - t2 - l2)

    # The series is created (specific part for each clusters)
    if cluster == 1:
        if l1 > 0:
            series[t1:(t1 + l1)] = us(l1, l1)
        if l2 > 0:
            series[t2:(t2 + l2)] = us(l2, l2)
        return series

    if cluster == 2:
        if l1 > 0:
            series[t1:(t1 + l1)] = us(l1, l1)
        if l2 > 0:
            series[t2:(t2 + l2)] = ds(l2, l2)
        return series

    if cluster == 3:
        if l1 > 0:
            series[t1:(t1 + l1)] = ds(l1, l1)
        if l2 > 0:
            series[t2:(t2 + l2)] = us(l2, l2)
        return series

    if cluster == 4:
        if l1 > 0:
            series[t1:(t1 + l1)] = ds(l1, l1)
        if l2 > 0:
            series[t2:(t2 + l2)] = ds(l2, l2)
        return series

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------


def two_patterns2(length_series, shift, warp, cluster, seed=None):

    np.random.seed(seed)

    # Upward shape
    def us(t, l):
        vector = 5 * np.ones((t,))
        vector[:int(np.ceil(l / 2.0))] *= -1
        return vector

    # Downward shape
    def ds(t, l):
        vector = 5 * np.ones((t,))
        vector[int(np.ceil(l / 2.0)):] *= -1
        return vector

    # The initial cut points are situated
    cut_point1 = np.floor(float(length_series) / 3)
    cut_point2 = np.floor(float(2 * length_series) / 3)

    # The cut point are moved depending on th shift and warp:
    t1 = int(np.max([cut_point1 + shift + warp, 1]))
    t2 = int(np.min([cut_point2 + shift + warp, length_series]))

    # The series is created (common part for all clusters)
    series = np.zeros((length_series,))
    # EXTRA NOISE.
    if t1 > 0:
        series[:t1] = np.random.randn(t1)
    if t2 < length_series:
        series[t2:] = np.random.randn(length_series - t2)

    # The series is created (specific part for each clusters)
    if cluster == 1:
        series[t1:t2] = us(t2 - t1, t2 - t1)
        return series

    if cluster == 2:
        series[t1:t2] = ds(t2 - t1, t2 - t1)
        return series

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------


def rational(length_series, shift, cluster, seed=None):

    np.random.seed(seed)

    # The shift is modified proportionally to fit the shape of the series.
    shift = float(40 * shift) / (length_series - 1)
    base = np.linspace(-20 + shift, 20 + shift, num=length_series)

    # The series is defined
    if (cluster == 1) or (cluster == 2):
        series = base / (base**2 + cluster)

    elif cluster == 3:
        import statsmodels.api as sm
        f = base / (base**2 + 1)
        lowess = sm.nonparametric.lowess
        series = lowess(f, base, frac=1. / 3)[:, 1]

    elif cluster == 4:
        f = base / (base**2 + 1)
        # The polynomial function is approximated by a cubic spline
        # Generating cubic spline
        spline = dmatrix('bs(base, df=20, degree=3, include_intercept=False)', {
                         'train': base}, return_type='dataframe')
        # Fitting Generalized linear model on transformed dataset
        splinelm = GLM(f, spline).fit()
        # Predictions on splines
        series = splinelm.predict()

    # The series is multiplied by a random number
    series *= np.random.uniform(low=1, high=1.1, size=1)
    return series

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------


def seasonal(length_series, shift, cluster, seed=None):

    np.random.seed(seed)

    base = np.arange(1, length_series + 1)
    # A basic period for the sinusoidal form is generated
    T = 0.5 * length_series
    core = 2 * np.pi * (base + shift) / T
    if cluster == 1:
        series = np.sin(core) + 0.5 * np.sin(2 * core - 2) + (1 / 3.0) * \
            (np.sin(2 * core - 2) + np.cos(4 * core - 2)) + np.sin(5 * core - 6)
    elif cluster == 2:
        series = 1.5 * np.sin(core) + 0.5 * np.cos(6 * core - 1) + \
            (1 / 6.0) * np.cos(2 * core - 7) + np.sin(2 * core - 9)
    elif cluster == 3:
        series = 1.2 * np.cos(3 * core + 7) + 0.9 * \
            np.sin(core - 2) + (1 / 3.0) * np.cos(core / 7.0)
    elif cluster == 4:
        series = 0.24 * np.sin(2.2 * core + 7) + 0.5 * \
            np.cos(core - 5) + (1 / 3.0) * np.cos(2 * core / 7.0)

    # The series is multiplied by a random number
    series *= np.random.uniform(low=1, high=1.3, size=1)
    return series
