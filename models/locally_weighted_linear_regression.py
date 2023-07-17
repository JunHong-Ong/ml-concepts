import numpy as np

from .base_model import BaseModel


class LocallyWeightedLinearRegression(BaseModel):

    def __init__(self, tau) -> None:
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Run gradient descent to minimize the cost function
        for Linear Regression.

        Args:
            x: The training example inputs (m, n).
            y: The training example labels (m, ).
        """
        self.x = x
        self.y = y

    def predict(self, x):
        """Make predictions given the inputs (x).

        Args:
            x : The inputs to the model (m, n).

        Returns:
            Returns the predicted values of the model (m, ).
        
        """
        m, n = x.shape
        y_pred = np.zeros(m)

        for i in range(m):
            W = np.diag(np.exp(-np.sum((self.x - x[i])**2, axis=1) / (2 * self.tau**2)))
            self.theta = np.linalg.inv(self.x.T.dot(W).dot(self.x)).dot(
                self.x.T.dot(W).dot(self.y))
            y_pred[i] = np.dot(x[i], self.theta)

        return y_pred
