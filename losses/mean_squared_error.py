import numpy as np


class MeanSquaredError():

    def __init__(self) -> None:
        pass

    def compute_cost(self, x, y, w, b):
        m = x.shape[0]
        y_pred = x.dot(w) + b
        y_pred = y_pred.squeeze()
        cost = 1/2 * 1/m * np.sum(np.power((y - y_pred), 2))

        return cost