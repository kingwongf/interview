import numpy as np
from scipy.optimize import minimize

def p_sig(w,sig):
    return w.dot(sig.dot(w.T))

def obj_func(w, params):
    sig, N = params
    return np.sum(np.square(w - p_sig(w,sig)**2/ sig.dot(w)*N))

def total_weight_constraint(x):
    return np.sum(x)-1.0

def long_only_constraint(x):
    return x


def w(cov):
    cons = ({'type': 'eq', 'fun': total_weight_constraint},
            {'type': 'ineq', 'fun': long_only_constraint})
    res = minimize(obj_func, w0, args=(cov, len(cov)), method='SLSQP', constraints=cons)