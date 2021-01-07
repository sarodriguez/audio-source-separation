spectrogram_type = 'complex'
n_fft = 1024
hop_length = 256
window = 'hamming'
window_length = n_fft
sample_time_frames = 128
# where the length of each time frame equals the hop length
time_frame_length = hop_length
total_audio_samples = (sample_time_frames - 1) * time_frame_length

import torch
from config import config
from datasets.musdb18_dataset import TrainMUSDB18Dataset
instruments = config.INSTRUMENTS
musdbwav_path = config.MUSDB18_WAV_PATH
sample_time_frames = config.SAMPLE_TIME_FRAMES
batch_size = config.BATCH_SIZE
prefetch_factor = config.PREFETCH_FACTOR
train_dataset = TrainMUSDB18Dataset(musdbwav_path, instruments, sample_length=sample_time_frames)
mus = train_dataset.mus
audio = mus.tracks[0].audio[:total_audio_samples]
audio_ch1 = audio[:, 0]
audio = torch.FloatTensor(audio)
hann = torch.hann_window(window_length)

# Simulate multiple audio by appending ina 3rd dimension
# Time, Ch, N
audios = torch.cat([audio.unsqueeze(dim=2), audio.unsqueeze(dim=2), audio.unsqueeze(dim=2)], dim=2)
initial_shape = audios.shape
# Time, Ch*N
audios_f = audios.flatten(1, 2)
# N, FreqBins, Time
spec_audios = torch.stft(audios_f.transpose(0, 1), n_fft, hop_length, window_length, hann, return_complex=True)
current_shape = spec_audios.shape
#


spec_comp = torch.stft(audio_ch1, n_fft, hop_length, window_length, hann, return_complex=True)
real_spec_comp = torch.view_as_real(spec_comp)
spec = torch.stft(audio_ch1, n_fft, hop_length, window_length, hann, return_complex=False)


comp_reconstructed = torch.istft(spec_comp, n_fft, hop_length, window_length, hann)
comp_reconstructed_2 = torch.istft(spec, n_fft, hop_length, window_length, hann)


spec_2ch = torch.stft(audio.transpose(0, 1), n_fft, hop_length, window_length, hann)#, return_complex=True)
comp_reconstructed_2ch = torch.istft(spec_2ch, n_fft, hop_length, window_length, hann)