import numpy as np

from modules.module import Module

"""
Implement **hyperbolic tangent** non-linearity (aka **Tanh**): 
Note that Tanh is scaled version of the sigmoid function.
"""


class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def updateOutput(self, inpt):
        # <Your Code Goes Here>
        self.output = np.tanh(inpt)
        # raise NotImplementedError()
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        # <Your Code Goes Here>
        self.gradInput = np.multiply(gradOutput, 1.0 - np.square(np.tanh(inpt)))
        # raise NotImplementedError()
        return self.gradInput

    def __repr__(self):
        return "Tanh"

