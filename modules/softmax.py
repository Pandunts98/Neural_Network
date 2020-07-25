import numpy as np

from modules.module import Module

"""
This one is probably the hardest but as others only takes 5 lines of code in total. 
- input:   **batch_size x n_feats**
- output: **batch_size x n_feats**
"""


class SoftMax(Module):
    def __init__(self):
        super(SoftMax, self).__init__()

    def updateOutput(self, inpt):
        # <Your Code Goes Here>
        z = np.subtract(inpt, inpt.max(axis=1, keepdims=True))
        self.output = np.divide(np.exp(z), np.sum(np.exp(z), axis=1, keepdims=True))
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        # <Your Code Goes Here>
        # raise NotImplementedError()
        sf = self.output
        m, n = inpt.shape
        sf_grad = np.zeros((m, n))
        for k in range(m):
            jac = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if j == i:
                        jac[i, j] = sf[k, j] * (1 - sf[k, j])
                    else:
                        jac[i, j] = -sf[k, i] * sf[k, j]
            sf_grad[k] = gradOutput[k].T @ jac
        self.gradInput = sf_grad
        return self.gradInput

    def __repr__(self):
        return "SoftMax"
