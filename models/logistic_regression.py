import numpy as np

from .base_model import BaseModel


class LogisticRegression(BaseModel):

    def fit(self, x, y):
        """Run gradient descent to minimize the cost function
        for Logistic Regression.

        Args:
            x: The training example inputs (m, n).
            y: The training example labels (m, ).
        """
        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros((n, ))
        if self.bias is None:
            self.bias = 0

        iteration = 1
        costs = []
        while iteration <= self.max_iter:
            theta_old = self.theta.copy()

            y_pred = self.predict(x)
            dw = 1/m * x.T.dot(y - y_pred)
            db = 1/m * np.sum(y - y_pred)
            self.theta += self.learning_rate * dw
            self.bias += self.learning_rate * db

            if np.linalg.norm(self.theta - theta_old, ord=1) < self.tol:
                break

            loss = self.loss(x, y)
            costs.append(loss)
            if self.verbose and iteration % 10 == 0:
                print(f"Loss on iteration {iteration}: {loss}")
            iteration += 1

        return costs
    
    def loss(self, x, y):
        """Compute the mean squared error for the model.

        Args:
            x: The training example inputs (m, n).
            y: The training example labels (m, ).
        """
        m, n = x.shape

        y_pred = self.predict(x)
        return 1/m * (y.T.dot(np.log(y_pred))) + ((1-y).dot(np.log(1-y_pred)))

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, x):
        """Make predictions given the inputs (x).

        Args:
            x : The inputs to the model (m, n).

        Returns:
            Returns the predicted values of the model (m, ).
        
        """
        return self._sigmoid(np.dot(x, self.theta) + self.bias)