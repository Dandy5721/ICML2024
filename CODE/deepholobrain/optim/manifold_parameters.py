from torch import nn


class StiefelParameter(nn.Parameter):
    """A parameter constrained to the Stiefel manifold (orthonormal columns)."""

    def __new__(cls, data=None, requires_grad=True):
        return super(StiefelParameter, cls).__new__(cls, data, requires_grad=requires_grad)

    def __repr__(self):
        return 'Parameter containing:' + self.data.__repr__()


class SPDParameter(nn.Parameter):
    """A parameter constrained to the manifold of SPD matrices."""

    def __new__(cls, data=None, requires_grad=True):
        return super(SPDParameter, cls).__new__(cls, data, requires_grad=requires_grad)

    def __repr__(self):
        return 'Parameter containing:' + self.data.__repr__()
