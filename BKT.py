from sklearn.base import BaseEstimator
import numpy as np
import itertools
import sys
from tqdm import tqdm

class BKT(BaseEstimator):
    def __init__(self, step = 0.1, bounded = True, best_k0 = True):
        self.k0 = 0.001
        self.transit = 0.001
        self.guess = 0.001
        self.slip = 0.001
        self.forget = 0.001

        self.k0_limit = 0.999
        self.transit_limit = 0.999
        self.guess_limit = 0.999
        self.slip_limit = 0.999
        self.forget_limit = 0.999

        self.step = step
        self.best_k0 = best_k0

        if bounded:
            self.k0_limit = 0.85
            self.transit_limit = 0.3
            self.guess_limit = 0.3
            self.slip_limit = 0.1

    def fit(self, X, y = None):

        if self.best_k0:
            self.k0 = self._find_k0(X)
            self.k0_limit = self.k0 + self.step

        k0s = np.arange(self.k0, self.k0_limit + self.step, self.step)
        transits = np.arange(self.transit, self.transit_limit + self.step, self.step)
        guesses = np.arange(self.guess, self.guess_limit + self.step, self.step)
        slips = np.arange(self.slip, self.slip_limit + self.step, self.step)

        all_parameters = [k0s, transits, guesses, slips]
        parameter_pairs = list(itertools.product(*all_parameters))

        min_error = sys.float_info.max
        for (k, t, g, s) in tqdm(parameter_pairs):
            error, _ = self._computer_error(X, k, t, g, s)
            if error < min_error:
                self.k0 = k
                self.transit = t
                self.guess = g
                self.slip = s
                min_error = error

        print "Traning RMSE: ", min_error
        return self.k0, self.transit, self.guess, self.slip

    def _computer_error(self, X, k, t, g, s):
        error = 0.0
        n = 0
        predictions = []

        for seq in X:
            current_pred = []
            pred = k
            for i, res in enumerate(seq):
                n += 1
                current_pred.append(pred)
                error += (res - pred) ** 2
                if res == 1.0:
                    p = k * (1 - s) / (k * (1 - s) + (1 - k) * g)
                else:
                    p = k * s / (k * s + (1 - k) * (1 - g))
                k = p + (1 - p) * t
                pred = k * (1 - s) + (1 - k) * g
                # pred = k
            predictions.append(current_pred)

        return (error / n)  ** 0.5, predictions

    def _find_k0(self, X):
        res = [seq[0] for seq in X]
        return np.mean(res)

    def predict(self, X):
        return self._computer_error(X, self.k0, self.transit, self.guess, self.slip)


if __name__ == "__main__":
    from sklearn.utils.estimator_checks import check_estimator
    # check_estimator(BKT)

    input_data = [
    [1,0,0,0,1,1,1],
    [0,0,0,1,0,0,1],
    [1,1,0,0,1,1]
    ]

    bkt = BKT(step = 0.01, best_k0 = True)
    bkt.fit(input_data)
    error, predictions =  bkt.predict([[0,0,0,0,0,0,1,1,1]])
    print error
    print predictions


