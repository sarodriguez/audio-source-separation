from torch import nn
from utils.functions import get_activation


class FullyConnectedControl(nn.Module):
    """
    Fully Connected control
    In our implementation we decided to move the gamma and beta creation into the FiLM calculation given that
    their dimensionalities depend on the type of FiLM being used. This control is basically an embedding created
    from a series of Fully connected layers
    """
    def __init__(self, input_depth: int, layer_depths: list, batch_norm=False, dropout=0.0,
                 bn_momentum=0.1, activation='relu'):
        super(FullyConnectedControl, self).__init__()
        input_depths = [input_depth] + layer_depths[:-1]
        self.control_layers = [FullyConnectedControlBlock(c_input_depth, c_output_depth, batch_norm, dropout, bn_momentum,
                                                          activation) for c_input_depth, c_output_depth in zip(input_depths,
                                                                                                  layer_depths)]
        self.seq_control_layers = nn.Sequential(*self.control_layers)

    def forward(self, X):
        return self.seq_control_layers(X)



class FullyConnectedControlBlock(nn.Module):
    def __init__(self, input_depth, output_depth, batch_norm, dropout, bn_momentum, activation):
        super(FullyConnectedControlBlock, self).__init__()
        components = list()
        components.append(nn.Linear(input_depth, output_depth))
        components.append(get_activation(activation_name=activation))
        if dropout:
            components.append(nn.Dropout(dropout))
        if batch_norm:
            components.append(nn.BatchNorm1d(num_features=output_depth, momentum=bn_momentum))
        self.control = nn.Sequential(*components)

    def forward(self, X):
        return self.control(X)


