from modules.module import Module
import numpy as np

"""
Implement **Rectified Linear Unit** non-linearity (aka **ReLU**): 
"""


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def updateOutput(self, inpt):
        # <Your Code Goes Here>
        self.output = np.maximum(inpt, 0)
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        # <Your Code Goes Here>
        self.gradInput = np.multiply(gradOutput, inpt > 0)
        return self.gradInput

    def __repr__(self):
        return "ReLU"
