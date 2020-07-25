import numpy as np

from modules.module import Module

"""
One of the most significant recent ideas that impacted NNs a lot is 
**Batch normalization**](http://arxiv.org/abs/1502.03167). The idea is simple,
yet effective: the features should be whitened ($mean = 0$, $std = 1$) all the 
way through NN. This improves the convergence for deep models letting it train
them for days but not weeks. **You are** to implement a part of the layer: mean
subtraction. That is, the module should calculate mean value for every feature
(every column) and subtract it.

Note, that you need to estimate the mean over the dataset to be able to predict
on test examples. The right way is to create a variable which will hold smoothed
mean over batches (exponential smoothing works good) and use it when forwarding
test examples.

When training calculate mean as folowing: 
```
    mean_to_subtract = self.old_mean * alpha + batch_mean * (1 - alpha)
```
and do backpropagation accordingly.

when evaluating (`self.training == False`) set $alpha = 1$.

- input:   **`batch_size x n_feats`**
- output: **`batch_size x n_feats`**
"""


class BatchMeanSubtraction(Module):
    def __init__(self, alpha=0.95):
        super(BatchMeanSubtraction, self).__init__()

        self.alpha = alpha
        self.old_mean = None
        self.old_var = None
        self.gamma = None
        self.beta = None
        self.batch_norm = None
        self.dgamma = None
        self.dbeta = None

    def updateOutput(self, inpt):
        # <Your Code Goes Here>
        mean = np.mean(inpt, axis=0, keepdims=True)
        var = np.var(inpt, axis=0, keepdims=True)

        if self.training:
            if self.old_mean is None:
                self.gamma = np.ones_like(var)
                self.beta = np.zeros_like(var)
                self.old_var = var
                self.old_mean = mean
            else:
                self.old_var = self.alpha * self.old_var + (1.0 - self.alpha) * var
                self.old_mean = self.alpha * self.old_mean + (1.0 - self.alpha) * mean
            self.batch_norm = (inpt - mean) / np.sqrt(var + 1e-8)
            self.output = self.gamma * self.batch_norm + self.beta
        else:
            batch_norm = (inpt - self.old_mean) / np.sqrt(self.old_var + 1e-8)
            self.output = self.gamma * batch_norm + self.beta
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        # <Your Code Goes Here>
        mean = np.mean(inpt, axis=0, keepdims=True)
        var = np.var(inpt, axis=0, keepdims=True)

        m = inpt.shape[0]
        X_mu = inpt - mean
        std_inv = 1. / np.sqrt(var + 1e-8)
        dX_norm = gradOutput * self.gamma
        dvar = np.sum(dX_norm * X_mu, axis=0) * -0.5 * std_inv ** 3
        dmu = np.sum(dX_norm * -std_inv, axis=0) + dvar * np.mean(-2. * X_mu, axis=0)

        dinpt = (dX_norm * std_inv) + (dvar * 2 * X_mu / m) + (dmu / m)
        self.dgamma = np.sum(gradOutput * self.batch_norm, axis=0)
        self.dbeta = np.sum(gradOutput, axis=0)
        self.gradInput = np.multiply(gradOutput, dinpt)
        return self.gradInput

    def getParameters(self):
        return [self.gamma.reshape(-1, ), self.beta.reshape(-1, )]

    def getGradParameters(self):
        return [self.dgamma, self.dbeta]

    def __repr__(self):
        return "BatchMeanNormalization"
