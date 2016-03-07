from sklearn.base import BaseEstimator
import numpy as np
import itertools

class BKT(BaseEstimator):
    def __init__(self, step = 0.1, bounded = True):
        self.k0 = 0.1
        self.learn = 0.2
        self.guess = 0.3
        self.slip = 0.4
        self.forget = 0.5

        self.k0_limit = 1.0
        self.learn_limit = 1.0
        self.guess_limit = 1.0
        self.slip_limit = 1.0
        self.forget_limit = 1.0

        self.step = step

        if bounded:
            self.k0_limit = 0.85
            self.learn_limit = 0.3
            self.guess_limit = 0.3
            self.slip_limit = 0.1

    def fit(self, X, y):

        k0s = np.arange(self.k0, self.k0_limit, self.step)
        learns = np.arange(self.learn, self.k0_limit, self.step)
        guesses = np.arange(self.guess, self.k0_limit, sef.step)
        slips = np.arange(self.slip, self.k0_limit, self.step)

        all_parameters = [k0s, learns, guesses, slips]
        parameter_pairs = list(itertools.product(*all_parameters))

        print len(parameter_pairs)

        for i in parameter_pairs:
            print i
            raw_input()




        return self

    def predict(self, X):
        return


if __name__ == "__main__":
    from sklearn.utils.estimator_checks import check_estimator

    bkt = BKT(False)
    bkt.fit(1,2)


