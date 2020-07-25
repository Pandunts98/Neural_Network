import numpy as np

from modules.module import Module

"""
Implement well-known **Sigmoid** non-linearity
"""


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def updateOutput(self, inpt):
        # <Your Code Goes Here>
        # raise NotImplementedError()
        self.output = np.divide(1, 1 + np.exp(-inpt))
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        # <Your Code Goes Here>
        # raise NotImplementedError()

        self.gradInput = np.multiply(gradOutput, np.multiply(self.output, 1 - self.output))
        return self.gradInput

    def __repr__(self):
        return "Sigmoid"
