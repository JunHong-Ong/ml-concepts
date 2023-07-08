import numpy as np


class GradientDescent():

    def __init__(self, learning_rate=0.01) -> None:
        self.lr = learning_rate

    def compute_gradient(self, x, y, w, b):
        m = y.shape[0]
        y_pred = x.dot(w) + b
        y_pred = y_pred.squeeze()
        
        dw = 1/m * (y - y_pred).dot(x)
        db = 1/m * np.sum(y - y_pred)

        w = w + (self.lr * dw)
        b = b + (self.lr * db)

        return w, b

# Batch, Mini-batch, Stochastic GD