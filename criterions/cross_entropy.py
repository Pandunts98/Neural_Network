from criterions.criterion import Criterion
import numpy as np

"""
You task is to implement the **CrossEntropyCriterion**. It should implement
multiclass log loss.
(http://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy)
Nevertheless there is a sum over `y` (target) in that formula, remember that 
targets are one-hot encoded. This fact simplifies the computations a lot. Note,
that criterions are the only places, where you divide by batch size. 
"""


class CrossEntropyCriterion(Criterion):
    def __init__(self):
        super(CrossEntropyCriterion, self).__init__()

    def updateOutput(self, inpt, target):
        # Use this trick to avoid numerical errors
        input_clamp = np.maximum(1e-15, np.minimum(inpt, 1 - 1e-15))
        # <Your Code Goes Here>
        L = -np.sum(np.multiply(target, np.log(input_clamp)), axis=1)
        self.output = L.mean()
        # raise NotImplementedError()
        return self.output

    def updateGradInput(self, inpt, target):
        # Use this trick to avoid numerical errors
        input_clamp = np.maximum(1e-15, np.minimum(inpt, 1 - 1e-15))
        # <Your Code Goes Here>
        self.gradInput = -np.multiply(np.divide(1, inpt.shape[0]), np.divide(target, input_clamp))
        # raise NotImplementedError()
        return self.gradInput

    def __repr__(self):
        return "CrossEntropyCriterion"
