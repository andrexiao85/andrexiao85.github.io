import torch

class LinearModel:

    def __init__(self):
        """
        self.w, torch.Tensor: the current weight vector w_k
        self.w_, torch.Tensor: the old weight vector w_{k-1}
        """
        self.w = None 
        self.w_ = None

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))
            self.w_ = torch.clone(self.w)

        # your computation here: compute the vector of scores s
        return X@self.w

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        return 1.0*(self.score(X) > 0)
    
class LogisticRegression(LinearModel):

    def loss(self, X, y):
        """
        Computes the gradient descent L(w) = 1/n * sum(i=1->n)[-y_i*log(sigma(s_i)) - (1-y_i)log(1-sigma(s_i))]
        where s_i = <w, x_i> and sigma(s_i)=1/(1+e^-s).

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}.
        """
        sigma = 1/(1+torch.exp(-self.score(X)))
        return (-y*sigma.log() - (1 - y)*(1-sigma).log()).mean()
    
    def grad(self, X, y):
        """
        Computes the gradient of the empirical risk L(w) using the feature matrix X 
        and target vector y.  
        """
        sigma = 1/(1+torch.exp(-self.score(X)))
        return ((sigma - y)[:, None] * X).mean(0)
    
    def hessian(self, X):
        sigma = 1/(1+torch.exp(-self.score(X)))
        D = (sigma*(1-sigma)).diag()
        return X.transpose(0, 1)@D@X
    
class GradientDescentOptimizer:

    def __init__(self, model):
        self.model = model

    def step(self, X, y, alpha = 0.01, beta = 0):
        """
        Compute one step of the logistic regression update using the feature matrix X 
        and target vector y. 
        """

        old_w = torch.clone(self.model.w)
        self.model.w += -alpha*self.model.grad(X, y) + beta*(self.model.w - self.model.w_)
        self.model.w_ = old_w

class NewtonOptimizer:

    def __init__(self, model):
        self.model = model

    def step(self, X, y, alpha = 0.01):
        self.model.w -= alpha*self.model.hessian(X).inverse()@self.model.grad(X, y)
