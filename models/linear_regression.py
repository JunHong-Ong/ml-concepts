import numpy as np

class LinearRegression():

    def __init__(self) -> None:
        self.w = None
        self.b = None

    def predict(self, x, w, b):
        """Make predictions using the inputs (x), weights (w) and bias (b).

        Args:
            x : shape=(m, n) The inputs to the model.
            w : shape=(n, 1) The weights parameter of the model.
            b : float        The bias of the model.

        Returns:
            Returns the predicted values of the model.
        
        """
        y_pred = x.dot(w) + b
        y_pred = y_pred.squeeze()

        return y_pred
    
    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss

    def fit(self, x, y, w, b, epochs=100):
        self.costs = []

        for _ in range(epochs):
            cost = self.loss.compute_cost(x, y, w, b)
            w, b = self.optimizer.compute_gradient(x, y, w, b)
            self.costs.append(cost)

        return w, b, self.costs
    
# Show diagnostic plots
#  - Loss
#  - Contour
#  - Data