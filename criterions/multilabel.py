from criterions.criterion import Criterion
import numpy as np

"""
**MultiLabelCriterion** for atribute classification, i.e. target is multiple-hot
encoded, could be multiple ones i.e. sample can be classified to more than one
classes.
"""


class MultiLabelCriterion(Criterion):
    def __init__(self):
        super(MultiLabelCriterion, self).__init__()

    def updateOutput(self, inpt, target):
        # Use this trick to avoid numerical errors
        input_clamp = np.maximum(1e-15, np.minimum(inpt, 1 - 1e-15))
        # <Your Code Goes Here>
        l = -np.sum((target * np.log(input_clamp) + (1 - target) * np.log(1 - input_clamp)), axis=1)
        self.output = l.mean()
        # raise NotImplementedError()
        return self.output

    def updateGradInput(self, inpt, target):
        # Use this trick to avoid numerical errors
        input_clamp = np.maximum(1e-15, np.minimum(inpt, 1 - 1e-15))

        # <Your Code Goes Here>
        # self.gradInput = -input_clamp + target
        self.gradInput = - ((target / input_clamp) - (1 - target) / (1 - input_clamp)) / inpt.shape[0]
        # raise NotImplementedError()
        return self.gradInput

    def __repr__(self):
        return "MultiLabelCriterion"
