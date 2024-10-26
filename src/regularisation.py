import numpy as np

class L1_Regularisation(object):
    
    @staticmethod
    def update_weight(ntrain, eta, lmbda, w_arr):
        return np.sign(w_arr) * eta * lmbda / float(ntrain)

    @staticmethod
    def get_cost(ntrain, lmbda, w_arr):
        return np.absolute(w_arr).sum() * lmbda / float(ntrain)


class L2_Regularisation(object):

    @staticmethod
    def update_weight(ntrain, eta, lmbda, w_arr):
        return w_arr * eta * lmbda / float(ntrain)

    @staticmethod
    def get_cost(ntrain, lmbda, w_arr):
        return 0.5 * lmbda * np.linalg.norm(w_arr)**2 / ntrain

