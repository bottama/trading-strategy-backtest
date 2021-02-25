""" Minimum Variance Portfolio """

import numpy as np
from scipy.optimize import minimize


def minimum_variance_portfolio(ret):

    # define objective function to minimize: variance
    def get_portfolio_variance(weights):
        weights = np.array(weights)  # check
        cov_mat = ret.cov()
        port_variance = np.dot(weights.T, np.dot(cov_mat, weights))
        return port_variance

    # equality constraint: sum of the weights = 1
    def weight_cons(weights):
        return np.sum(weights) - 1

    # model set-up
    # - long only portfolio
    # - initial guess
    # - constraints
    bounds_lim = ((0, 1),) * len(ret.columns)
    init_weights = [1 / len(ret.columns)] * len(ret.columns)
    constraint = {'type': 'eq', 'fun': weight_cons}

    # find optimal portfolio
    opt_port = minimize(fun=get_portfolio_variance,
                        x0=init_weights,
                        bounds=bounds_lim,
                        constraints=constraint,
                        method='SLSQP')

    # find optimal weights
    opt_weights = list(opt_port['x'])

    return opt_weights
