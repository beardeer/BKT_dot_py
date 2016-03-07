from sklearn.base import BaseEstimator
import numpy as np
import itertools
import sys

class BKT(BaseEstimator):
    def __init__(self, input_data, large_step = 0.1, small_step = 0.01, bounded = True):
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

    def fit(self, X, y = None):

        k0s = np.arange(self.k0, self.k0_limit, self.large_step)
        learns = np.arange(self.learn, self.k0_limit, self.large_step)
        guesses = np.arange(self.guess, self.k0_limit, self.large_step)
        slips = np.arange(self.slip, self.k0_limit, self.large_step)

        all_parameters = [k0s, learns, guesses, slips]
        parameter_pairs = list(itertools.product(*all_parameters))

        min_error = sys.float_info.max
        for (k, l, g, s) in parameter_pairs:
            error = self.computer_error(X, k, l, g, s)

            if error < min_error:
                self.k0 = k
                self.learn = l
                self.guess = g
                self.slip = s

                min_error = error

        return self.k0, self.learn, self.guess, self.slip

    def computer_error(self, X, k, l, g, s):
        error = 0.0

        for seq in X:
            for i, n in enumerate(seq):
                error += (n - pred) ** 2
                if i == 0:
                    pred = k
                else:
                    p_c = k * (1 - s) / k * (1 - s) + (1 - k) * k
                    p_ic = k * s / (k * s + (1 - k) * (1 - g))
                    if n == 1.0:
                        k = p_c + (1 - p_ic) * l
                    else:
                        k = p_ic + (1 - p_ic) * l
                    pred = k
        return error

    def load_data(self):
        pass

    def predict(self, X):
        return


if __name__ == "__main__":
    from sklearn.utils.estimator_checks import check_estimator

    input_data = [
    [1,0,0,0,1,1,1],
    [0,0,0,1,0,0,1],
    [1,1,0,0,1,1]
    ]

    bkt = BKT(input_data, bounded = False)
    print bkt.fit(input_data)


