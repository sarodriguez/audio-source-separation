import torch
from utils.functions import get_stft_window


class Spectrogramer:
    def __init__(self, spectrogram_type, n_fft, hop_length, window, window_length, device):
        assert spectrogram_type in ('complex', 'magnitude')
        self.spectrogram_type = spectrogram_type
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        self.window_length = window_length
        # TODO should the (hamming) window be a trainable parameter?
        self.stft_window = get_stft_window(window, window_length).to(device)

    def get_spectrogram(self, audio):
        audio, initial_shape = self.reshape_multichannel(audio)
        if self.spectrogram_type == 'complex':
            spec = torch.stft(audio, self.n_fft, self.hop_length, window=self.stft_window, return_complex=True)
            spec = self.undo_reshape_multichannel(spec, initial_shape=initial_shape)
            phase = None

        # Magnitude and Phase spectrogram
        else:
            spec = torch.stft(audio, self.n_fft, self.hop_length, window=self.stft_window, return_complex=False)
            phase = spec[:, :, :, 1]
            phase = self.undo_reshape_multichannel(phase, initial_shape)
            spec = spec[:, :, :, 0]
            spec = self.undo_reshape_multichannel(spec, initial_shape=initial_shape)

        return spec, phase

    def reconstruct(self, spectrogram, phase):
        phase, initial_shape = self.reshape_multichannel_reconstruct(phase)
        spectrogram, initial_shape = self.reshape_multichannel_reconstruct(spectrogram)
        if self.spectrogram_type == 'complex':
            audio = torch.istft(spectrogram, self.n_fft, self.hop_length, window=self.stft_window, return_complex=True)
            audio = self.undo_reshape_multichannel_reconstruct(audio, initial_shape)

        # Magnitude and Phase spectrogram
        else:
            complex_spec = torch.cat([spectrogram.unsqueeze(-1), phase.unsqueeze(-1)], dim=-1)
            complex_spec = torch.view_as_complex(complex_spec)
            audio = torch.istft(complex_spec, self.n_fft, self.hop_length, window=self.stft_window, return_complex=False)
            audio = self.undo_reshape_multichannel_reconstruct(audio, initial_shape)

        return audio

    def reshape_multichannel_reconstruct(self, spec):
        if len(spec.shape) == 4:
            initial_shape = spec.shape
            # The channel dimension is considered to be the 2nd dimension by default
            spec = torch.flatten(spec, start_dim=0, end_dim=1)
            # spec = torch.flatten(spec.permute(1, 2, 0, 3), start_dim=2, end_dim=3).permute(2, 0, 1)
            return spec, initial_shape
        else:
            return spec, None

    def undo_reshape_multichannel_reconstruct(self, audio, initial_shape=None):
        return self.undo_reshape_multichannel(audio, initial_shape=initial_shape)

    def reshape_multichannel(self, audio):
        if len(audio.shape) == 3:
            initial_shape = audio.shape
            # The channel dimension is considered to be the 2nd dimension by default
            audio = torch.flatten(audio, start_dim=0, end_dim=1)
            # audio = torch.flatten(audio.transpose(0, 1), start_dim=1, end_dim=2).transpose(0, 1)
            return audio, initial_shape
        else:
            return audio, None

    def undo_reshape_multichannel(self, spec, initial_shape=None):
        if initial_shape is not None:
            # initial shape: bs, #channels, clip_length
            current_shape = spec.shape
            # current shape: bs*#channels, nfft, time
            # Returned shape should be: bs, #channels, nfft, time,
            final_shape = initial_shape[:2] + current_shape[1:]
            spec = torch.reshape(spec, final_shape)

            # # initial shape: bs, clip_length, #channels
            # current_shape = spec.shape
            # # current shape: bs*#channels, nfft, time
            # # Intermediate shape should be: bs, #channels, nfft, time,
            # final_shape = initial_shape[:1] + initial_shape[-1:] + current_shape[1:]
            # spec = torch.reshape(spec, final_shape)
            # # The returned tensor  will have the following dimension order: bs, nfft, time, #channels
            # spec_dimensions = len(current_shape[1:])
            # spec_dimensions_list = [i+2 for i in range(spec_dimensions)]
            # spec_dimensions_list = [0] + spec_dimensions_list + [1]
            # spec = spec.permute(*spec_dimensions_list)
            return spec
        else:
            return spec


