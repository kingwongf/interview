from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv,pinv
from scipy.special import binom
from scipy.optimize import minimize

 # risk budgeting optimization
def calculate_portfolio_var(w,V):
    # function that calculates portfolio risk
    w = np.matrix(w)
    # w = np.array(w)
    return (w*V*w.T)[0,0]

def calculate_risk_contribution(w,V):
    # function that calculates asset contribution to total risk
    w = np.matrix(w)
    sigma = np.sqrt(calculate_portfolio_var(w,V))
    # Marginal Risk Contribution
    MRC = V*w.T
    # Risk Contribution
    RC = np.multiply(MRC,w.T)/sigma
    return RC

def risk_budget_objective(x,pars):
    # calculate portfolio risk
    V = pars[0]# covariance table
    x_t = pars[1] # risk target in percent of portfolio risk
    sig_p =  np.sqrt(calculate_portfolio_var(x,V)) # portfolio sigma
    risk_target = np.asmatrix(np.multiply(sig_p,x_t))
    asset_RC = calculate_risk_contribution(x,V)
    J = sum(np.square(asset_RC-risk_target.T))[0,0] # sum of squared error
    return J

def total_weight_constraint(x):
    return np.sum(x)-1.0

def long_only_constraint(x):
    return x




def risk_parity_weighting(vcv_matrix, risk_budget='equal'):

    if risk_budget=='equal':
        risk_budget = np.array([1/len(vcv_matrix)]*len(vcv_matrix))
    # w0 = np.array([1 / len(vcv_matrix)] * len(vcv_matrix))

    n = len(vcv_matrix)
    # w0 = np.random.random(size=(len(vcv_matrix)))
    #
    # if np.sum(w0) !=1:
    #     r = 1 - np.sum(w0)
    #     w0 = w0- r/len(w0)

    ## using binomial distribution to initalise weights
    w0 = np.array(list(binom(n, x) * (0.2 ** x) * ((0.8) ** (n - x)) for x in range(0, n)))


    cons = ({'type': 'eq', 'fun': total_weight_constraint},
            {'type': 'ineq', 'fun': long_only_constraint})

    res = minimize(risk_budget_objective, w0, args=[vcv_matrix, risk_budget],
                   method='SLSQP', constraints=cons, options={'disp': False},
                   bounds=tuple((0,1) for _ in range(0,len(vcv_matrix))),
                   tol=1e-4)
    return np.array(res.x)


# x_t = [0.25, 0.25, 0.25, 0.25] # your risk budget percent of total portfolio risk (equal risk)


# V = np.matrix('123 37.5 70 30; 37.5 122 72 13.5; 70 72 321 -32; 30 13.5 -32 52')/100  # covariance
# #
# #
# # w0 = np.array([1/len(x_t)]*len(x_t))
# # cons = ({'type': 'eq', 'fun': total_weight_constraint},
# #         {'type': 'ineq', 'fun': long_only_constraint})
#
#
# print(risk_parity_weighting(V, 'equal'))
# res= minimize(risk_budget_objective, w0, args=[V,x_t], method='SLSQP',constraints=cons, options={'disp': True})
# w_rb = np.asmatrix(res.x)


# print(w_rb)