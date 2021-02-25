""" Mean-Variance Portfolio """

# import modules
import numpy as np
from scipy.optimize import minimize


def mean_variance_portfolio(ret):

    # define objective function to minimize: sharpe ratio
    def get_portfolio_sr(weights):

        weights = np.array(weights)  # check

        # expected returns
        port_ret = np.dot(ret, weights)
        mean_ret = port_ret.mean()

        # volatility
        cov_mat = ret.cov()
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))

        # sharpe ratio
        port_sr = mean_ret / port_std
        return port_sr

    def objective_fun(weights):
        neg_sr = get_portfolio_sr(weights) * (-1)
        return neg_sr

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
    opt_port = minimize(fun=objective_fun,
                        x0=init_weights,
                        bounds=bounds_lim,
                        constraints=constraint,
                        method='SLSQP')

    # find optimal weights
    opt_weights = list(opt_port['x'])

    return opt_weights