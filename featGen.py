import numpy as np
from scipy.stats.mstats import zscore, winsorize
import pandas as pd
import statsmodels.api as sm


def func_Clayton(u, v, kendall_corr):
    '''
    C(u,v|θ) = (θ+1)(u^−θ+v^−θn−1)^(−2−1/θ)u^(−θ−1)v^(−θ−1)

    :param u: U ~ (0,1)
    :param v: V ~ (0,1)
    :param kendall_corr: Kendall's rank correlation
    :return: C(u,v|θ, u, v), density function
    '''
    theta = 2*kendall_corr /(1 -kendall_corr)
    return (1+theta) * (u**-theta + v**-theta - 1)**(-2 - 1/theta) * (u**(-theta -1)) *(v**(-theta-1))

def partial_derivative_Clayton(u, v, kendall_corr):
    '''
    C(v∣u)=u^(−θ−1)(u^−θ+v^−θ −1)^(−1/θ−1)
    :param u:
    :param v:
    :param kendall_corr:
    :return: C(v∣u) => P(v|u)
    '''
    theta = 2 * kendall_corr / (1 - kendall_corr)
    return u**(-theta-1) * (u**-theta + v**-theta - 1)**(-1/theta -1 )


