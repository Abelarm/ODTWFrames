def sim_arima(model, n, errors=None, nstart=None, seed=123456):
    """
    Simulate from an ARIMA(p,d,q) model.

    Parameters
    ----------
    model: list of tuples, where model[1]=(p,d,q) values and model[1,2] the corresponding AR and/or MA coefficients.
            If model[0]=(0,0,0) then model[1] and model[2] must be None.
    n: length of output series, before un-differencing integer.
    errors: an optional times series of errors. If not provided, randn is pip3used.
    nstart:	length of burn-in period. If NA, the default, a reasonable value is computed.
    """

    from numpy.polynomial import polynomial as poly
    from statsmodels.tsa.filters.filtertools import recursive_filter, convolution_filter
    import pandas as pd
    import numpy as np
    import sys

    if seed is not None:
        np.random.seed(seed)

    # Initialization
    p, d, q = model[0]
    AR = list(model[1])
    MA = list(model[2])

    if p > 0:
        minroots = np.min(abs(poly.polyroots([1] + [-i for i in AR])))
        if minroots <= 1:
            sys.exit('AR part of the model is not stationary')

    if p != len(AR):
        sys.exit('incosistent specification of AR order')
    if q != len(MA):
        sys.exit('incosistent specifiaction of MA order')

    if nstart is None:
        nstart = p + q + (np.ceil(6.0 / np.log(minroots)) if p > 0 else 0)

    if errors is None:
        errors = np.random.randn(n)
    if errors.size < n:
        sys.exist('errors.size  must be larger or equal to n')

    x = pd.Series(
        data=np.concatenate((np.random.randn(nstart), errors[:n])),
        index=range(-nstart, n))

    if d > 0 and x.size > d:
        for i in range(d):
            x = x.diff(1)
    elif x.size <= d:
        sys.exit('number of differences must be smaller n')
    if q > 0:
        x = convolution_filter(x, filt=MA, nsides=1)
        x[:q] = 0
    if p > 0:
        x = recursive_filter(x, ar_coeff=AR)
    if nstart > 0:
        x = x[:-nstart]
    if d > 0:
        for i in range(d, 0, -1):
            x[d - 1] = 0
            x = x.cumsum()
    return x
