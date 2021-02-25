""" Equally weighted Portfolio """


def equally_weighted_portfolio(ret):
    init_weights = [1 / len(ret.columns)] * len(ret.columns)
    opt_weights = init_weights

    return opt_weights
