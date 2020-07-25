import numpy as np

from modules.module import Module

"""
Implement **dropout**(https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf).
The idea and implementation is really simple: just multimply the input 
by Bernoulli(p) mask. 

This is a very cool regularizer. In fact, when you see your net is overfitting
try to add more dropout. It is hard to test, since every `forward` requires 
sampling a new mask, that is the only reason we need `fix_mask` parameter
in there. 

While training (`self.training == True`) it should sample a mask on each 
iteration (for every batch). When testing this module should implement identity
transform i.e. `self.output = input * p`.

- input:   **`batch_size x n_feats`**
- output: **`batch_size x n_feats`**
"""


class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()

        self.p = p
        self.mask = None

    def updateOutput(self, inpt):
        # <Your Code Goes Here>
        if self.training:
            self.mask = np.divide(np.random.binomial(1, self.p, size=inpt.shape), self.p)
            self.output = np.multiply(inpt, self.mask)
        else:
            self.output = inpt
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        # <Your Code Goes Here>
        # raise NotImplementedError()
        self.gradInput = np.multiply(gradOutput, self.mask)
        return self.gradInput

    def __repr__(self):
        return "Dropout"
