from torch import nn

class FiLM(nn.Module):
    def __init__(self):
        super(FiLM, self).__init__()

    def forward(self, X, z):
        """

        :param z: Conditioning/Control vector containing the instrument
        :param X: Feature maps
        :return:
        """
        return (self.gamma(z).T*X.T + self.beta(z).T).T

    @staticmethod
    def get_film(film_type, control_depth, channels):
        if film_type == 'complex':
            return ComplexFiLM(control_depth, channels)
        elif film_type == 'simple':
            return SimpleFiLM(control_depth)


class SimpleFiLM(FiLM):
    def __init__(self, control_depth):
        super(SimpleFiLM, self).__init__()
        self.gamma = nn.Linear(control_depth, 1)
        self.beta = nn.Linear(control_depth, 1)


class ComplexFiLM(FiLM):
    def __init__(self, control_depth, channels):
        super(ComplexFiLM, self).__init__()
        self.gamma = nn.Linear(control_depth, channels)
        self.beta = nn.Linear(control_depth, channels)

