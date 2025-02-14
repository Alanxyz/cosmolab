import numpy as np

def getmodel(alpha, beta, mb1, dm):

    mb = lambda mstellar: mb1 if mstellar < 10**10 else mb1 + dm
    mbarr = lambda mbstellar: np.array([ mb(m) for m in mbstellar ])

    mu = lambda mbstar, shape, color: mbstar - (mbarr(mbstar) - alpha * shape + beta * color)
    return mu
