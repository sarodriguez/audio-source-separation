from torch import nn
from models.mechanisms.control import FullyConnectedControl
from utils.functions import get_activation
from models.mechanisms.film import FiLM
from models.mechanisms.film import SimpleFiLM, ComplexFiLM
import torch

class ConditionedUNet(nn.Module):
    def __init__(self,
                 n_instruments,
                 n_input_audio_channels,
                 encoderdecoder_channels,
                 encoder_dropout,
                 encoder_batchnorm,
                 encoder_bn_momentum,
                 encoder_activation,
                 encoder_filter_size,
                 encoder_stride,
                 encoder_padding,
                 decoder_dropout,
                 decoder_batchnorm,
                 decoder_bn_momentum,
                 decoder_activations,
                 decoder_filter_size,
                 decoder_stride,
                 decoder_padding,
                 decoders_with_skip_connection,
                 film_type,


                 control_type,
                 control_layer_depths=None,
                 control_layer_batchnorm=None,
                 control_layer_dropout=None,
                 control_layer_bn_momentum=None,
                 control_layer_activation=None,
                 control_embedding_dimension=None

                 ):

        super(ConditionedUNet, self).__init__()
        # Defining the control for FiLM
        if control_type == 'fully_connected':
            self.control = FullyConnectedControl(n_instruments, control_layer_depths, control_layer_batchnorm,
                                                 control_layer_dropout, control_layer_bn_momentum,
                                                 control_layer_activation)
        elif control_type == 'embedding':
            self.control = nn.Embedding(n_instruments, embedding_dim=control_embedding_dimension)
        else:
            raise NotImplementedError

        # Defining the encoder components
        encoder_in_channels = [n_input_audio_channels] + encoderdecoder_channels[:-1]
        # Note that we use the last element from the control layer depths since this is the depth received by FiLm
        self.encoder = nn.ModuleList([EncoderBlock(in_channels, out_channels, encoder_dropout, encoder_batchnorm,
                                                   encoder_bn_momentum, encoder_activation, encoder_filter_size,
                                                   encoder_stride, encoder_padding, control_layer_depths[-1],
                                                   film_type) for in_channels,
                                                                  out_channels in zip(encoder_in_channels,
                                                                                      encoderdecoder_channels)])

        # Defining the decoder components
        self.decoder = nn.ModuleList()
        # Note that we invert the channels in and out for the decoder
        # We also add the skip connection flag in order to know if we should, or not, be using a skip connection
        # We do a similar thing for the activations, where we can have different activations for each decoder layer
        for (in_channels, out_channels, activation, skip_connection) in zip(encoderdecoder_channels[::-1],
                                                                            encoder_in_channels[::-1],
                                                                            decoder_activations,
                                                                            decoders_with_skip_connection):
            decoder_block = DecoderBlock(in_channels, out_channels, decoder_dropout, decoder_batchnorm,
                                         decoder_bn_momentum, activation, decoder_filter_size, decoder_stride,
                                         decoder_padding, skip_connection)
            self.decoder.append(decoder_block)


    def forward(self, X, condition):
        # First we go through the control mechanism, to create an 'embedding' of the input condition(source)
        z = self.control(condition)
        # Then we iterate over the different encoder layers
        encoder_outputs = list()
        for encoder_block in self.encoder:
            X = encoder_block(X, z)
            encoder_outputs.append(X)

        # The encoder outputs are then sent into the decoder
        for skip_connection_input, decoder_block in zip(encoder_outputs[::-1], self.decoder):
            # Blocks with / without skip connection are handled by the DecoderConnection object
            X = decoder_block(X, skip_connection_input)

        return X


class EncoderBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout,
                 batchnorm,
                 bn_momentum,
                 activation,
                 filter_size,
                 stride,
                 padding,
                 control_depth,
                 film_type):
        """

        """
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=filter_size, stride=stride, padding=padding)
        if batchnorm:
            self.batchnorm = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        else:
            self.batchnorm = nn.Identity()
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()
        self.activation = get_activation(activation_name=activation)

        self.film = FiLM.get_film(film_type, control_depth, out_channels)

    def forward(self, X, z):
        X = self.conv(X)
        X = self.batchnorm(X)
        X = self.film(X, z)
        X = self.activation(X)
        return X


class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout,
                 batchnorm,
                 bn_momentum,
                 activation,
                 filter_size,
                 stride,
                 padding,
                 skip_connection):
        super(DecoderBlock, self).__init__()
        if skip_connection:
            self.conv = nn.ConvTranspose2d(2 * in_channels, out_channels, kernel_size=filter_size, stride=stride,
                                           padding=padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=filter_size, stride=stride,
                                           padding=padding)
        if batchnorm:
            self.batchnorm = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        else:
            self.batchnorm = nn.Identity()
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()
        self.activation = get_activation(activation_name=activation)
        self.skip_connection = skip_connection

    def forward(self, X, X_skip):
        if self.skip_connection:
            X = torch.cat([X, X_skip], dim=1)
        X = self.conv(X)
        X = self.batchnorm(X)
        X = self.activation(X)
        return X

