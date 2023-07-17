class BaseModel():

    def __init__(self,
                 theta=None,
                 bias=None,
                 learning_rate=0.2,
                 max_iter=100,
                 tol=1e-5,
                 verbose=True) -> None:
        self.theta = theta
        self.bias = bias
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def fit(self, x, y):
        NotImplementedError()

    def predict(self, x):
        NotImplementedError()

# Show diagnostic plots
#  - Loss
#  - Contour
#  - Data