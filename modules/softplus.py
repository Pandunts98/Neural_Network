import numpy as np

from modules.module import Module

"""
Implement **SoftPlus**
(https://en.wikipedia.org/wiki%2FRectifier_%28neural_networks%29) activations.
Look, how they look a lot like ReLU.
"""


class SoftPlus(Module):
    def __init__(self):
        super(SoftPlus, self).__init__()

    def updateOutput(self, inpt):
        # <Your Code Goes Here>
        self.output = np.log(1 + np.exp(inpt))
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        # <Your Code Goes Here>
        self.gradInput = np.multiply(gradOutput, np.divide(1, 1 + np.exp(-inpt)))
        return self.gradInput

    def __repr__(self):
        return "SoftPlus"
