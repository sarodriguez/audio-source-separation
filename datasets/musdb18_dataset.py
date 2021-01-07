from torch.utils.data import Dataset
import musdb
from torch.utils.data.dataset import T_co
import torch
import numpy as np

import soundfile

class MUSDB18Dataset(Dataset):

    def __init__(self, musdbwav_path, subset_split, instruments, sample_length):
        assert subset_split in ('train', 'test', 'valid')
        if subset_split in ('train', 'valid'):
            self.mus = musdb.DB(root=musdbwav_path, subsets='train', split=subset_split, is_wav=True)
        elif subset_split == 'test':
            self.mus = musdb.DB(root=musdbwav_path, subsets='test', is_wav=True)
        self.sample_length = sample_length
        self.instruments = instruments

    def __len__(self):
        return len(self.mus)


class TrainMUSDB18Dataset(MUSDB18Dataset):
    def __init__(self, musdbwav_path, instruments, sample_length):
        subset_split = 'train'
        MUSDB18Dataset.__init__(self, musdbwav_path, subset_split, instruments, sample_length)
        self.total_chunks = int(sum([np.ceil(len(track)/self.sample_length) for track in self.mus.tracks]))

    def __len__(self):
        return self.total_chunks

    def __getitem__(self, index) -> T_co:
        # We get a random instrument from the list of instruments, which is going to be target
        rand_inst_index = np.random.choice(np.arange(len(self.instruments)))
        instrument_ohe = [0] * len(self.instruments)
        instrument_ohe[rand_inst_index] = 1

        # We also get random tracks for each of the instruments
        rand_tracks = [self.get_random_audio(instrument) for instrument in self.instruments]
        # We sum all the tracks, creating a new mixture
        mixture_track = sum(rand_tracks)
        # We assign the target track
        target_track = rand_tracks[rand_inst_index]

        # Transpose both tracks, so we have a final shape of: (ch, time)
        return torch.FloatTensor(mixture_track.T), torch.FloatTensor(target_track.T), torch.FloatTensor(instrument_ohe)
        
    def get_random_audio(self, instrument):
        """
        Get a random audio array, following the dataset's sample length, for the given instrument
        :param instrument:
        :return: np.array with the raw audio waveforms
        """
        # We get a random track
        rand_track = np.random.randint(0, len(self.mus))
        # Then a random starting position within this random track
        rand_start = np.random.randint(0, len(self.mus[rand_track]) - self.sample_length + 1)

        audio, sample_rate = soundfile.read(self.mus.tracks[rand_track].sources[instrument].path, start=rand_start,
                                            stop=rand_start + self.sample_length)
        return audio


class TestMUSDB18Dataset(MUSDB18Dataset):
    def __init__(self, musdbwav_path, instruments, sample_length, subset_split):
        assert subset_split in ('test', 'valid')
        MUSDB18Dataset.__init__(self, musdbwav_path, subset_split, instruments, sample_length)
        self.track_lengths = [len(track) for track in self.mus.tracks]
        self.n_instruments = len(self.instruments)
        self.chunks_per_instrument = np.ceil(np.array(self.track_lengths)/sample_length)
        self.chunks_per_track = self.chunks_per_instrument*self.n_instruments
        self.cum_chunks = np.cumsum(self.chunks_per_track)
        # We also initialize the 'mapper' from id to track + offset within a track

    def get_instrument_track_and_offset(self, chunk_index):
        # Chunk index is a single number that should be able to index across all the different instruments for
        # all the tracks.
        for i, cumulative_chunks in enumerate(self.cum_chunks):
            if chunk_index < cumulative_chunks:
                # This is the index of the current song/track in the mus db
                track_index = i
                # This is the total cumulative song chunks for the songs/tracks before the current one
                previous_cum_chunks = 0 if i == 0 else self.cum_chunks[i-1]
                # This is the offset within the current track, given the elapsed chunks
                track_chunk_index = (chunk_index - previous_cum_chunks)
                instrument_index = int(track_chunk_index // self.chunks_per_instrument[i])
                track_offset = int((track_chunk_index % self.chunks_per_instrument[i]) * self.sample_length)
                return track_index, instrument_index, track_offset

    def __len__(self):
        """
        Returns the total amount of chunks up until the last track, which is the length in chunks of the dataset
        :return:
        """
        return int(self.cum_chunks[-1])

    def __getitem__(self, index) -> T_co:
        # We get the track_index, instrument and track offset for the received index
        track_index, instrument_index, track_offset = self.get_instrument_track_and_offset(index)
        instrument_ohe = [0] * len(self.instruments)
        instrument_ohe[instrument_index] = 1

        # Retrieve the array that contains the audio
        instrument_name = self.instruments[instrument_index]
        target_audio_chunk = self.mus.tracks[track_index].targets[instrument_name].audio[track_offset:
                                                                                          track_offset +
                                                                                          self.sample_length]
        mixture_audio_chunk = self.mus.tracks[track_index].targets['linear_mixture'].audio[track_offset:
                                                                                           track_offset +
                                                                                           self.sample_length]
        # If the chunk is smaller than the pre established sample_length, then we pad the array with zeros
        # until it matches the size of sample_length
        if (target_audio_chunk.shape[0] != self.sample_length) and (mixture_audio_chunk.shape[0] != self.sample_length):
            target_audio_chunk = np.append(target_audio_chunk,
                                           np.zeros((self.sample_length - target_audio_chunk.shape[0],
                                                     target_audio_chunk.shape[1])), axis=0)
            mixture_audio_chunk = np.append(mixture_audio_chunk,
                                            np.zeros((self.sample_length - mixture_audio_chunk.shape[0],
                                                      mixture_audio_chunk.shape[1])), axis=0)

        # Transpose both tracks, so we have a final shape of: (ch, time)
        return torch.FloatTensor(mixture_audio_chunk.T), torch.FloatTensor(target_audio_chunk.T), \
               torch.FloatTensor(instrument_ohe), track_index, track_offset

    def get_random_audio(self, instrument):
        """
        Get a random audio array, following the dataset's sample length, for the given instrument
        :param instrument:
        :return: np.array with the raw audio waveforms
        """
        # We get a random track
        rand_track = np.random.randint(0, len(self))
        # Then a random starting position within this random track
        rand_start = np.random.randint(0, len(self.mus[rand_track]) - self.sample_length + 1)

        audio, sample_rate = soundfile.read(self.mus.tracks[rand_track].sources[instrument].path, start=rand_start,
                                            stop=rand_start + self.sample_length)
        return audio

