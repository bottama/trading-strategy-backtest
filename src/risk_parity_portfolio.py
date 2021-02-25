""" Risk parity portfolio """


def risk_parity_portfolio(ret):
    init_guess = 1 / ret.std()
    opt_weights = list(init_guess / init_guess.sum())

    return opt_weights
