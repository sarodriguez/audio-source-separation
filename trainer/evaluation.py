import torch
import museval

class Evaluator:
    def __init__(self, model, spectrogramer, loss_function, data_loader, musdb_dataset, instruments, device):
        self.model = model.to(device)
        self.spectrogramer = spectrogramer
        self.loss_function = loss_function
        self.validation_loader = data_loader
        self.losses = []
        self.accuracies = []
        self.musdb_dataset = musdb_dataset
        self.musdb18evaluator = MUSDB18Evaluator(musdb_dataset.mus, instruments)
        self.device = device

    def evaluate(self):
        # Set the model in evaluation mode
        self.model.eval()
        with torch.no_grad():
            losses = list()
            for i, data in enumerate(self.validation_loader, 0):
                # Here the inputs are the mixture track, targets are the final track and condition is the OHE instrument
                inputs, target, condition, track_index, track_offset = data
                # Send to the correct device
                inputs, target, condition = inputs.to(self.device), target.to(self.device), condition.to(self.device)
                # We get the phase, on a complex spec we would receive None for the phase
                inputs, inp_phase = self.spectrogramer.get_spectrogram(inputs)
                target, targ_phase = self.spectrogramer.get_spectrogram(target)
                outputs = self.model(inputs, condition)
                outputs = outputs * inputs
                loss = self.loss_function(outputs, target)
                outputs = self.spectrogramer.reconstruct(outputs, targ_phase)
                self.musdb18evaluator.add_new_predictions(outputs, track_index, condition)
                losses.append(loss.item())
            # accumulate the losses and average the accuracy.
            cum_loss = sum(losses)
            self.losses.append(cum_loss)
        self.model.train()
        return cum_loss, self.musdb18evaluator.results.agg_frames_tracks_scores(), \
               self.musdb18evaluator.results.agg_frames_scores()

class MUSDB18Evaluator:
    def __init__(self, mus_db, instruments):
        self.current_output = dict()
        self.instruments = instruments
        self.current_track = None
        self.current_instrument = None
        self.current_instrument_ohe = None
        self.mus_db = mus_db
        self.reset_current_results()
        self.results = museval.EvalStore(frames_agg='median', tracks_agg='median')

    def reset_current_results(self):
        for instrument in self.instruments:
            self.current_output[instrument] = list()
        self.current_track, self.current_instrument, self.current_instrument_ohe = None, None, None

    def get_current_output_length(self):
        return sum(len(output) for output in self.current_output.values())

    def get_results_df(self):
        return self.results.df

    def update_current_track(self, track, instrument_ohe, instrument_label=None):
        self.current_track = track
        self.update_current_instrument(instrument_ohe, instrument_label)

    def update_current_instrument(self, instrument_ohe, instrument_label=None):
        self.current_instrument_ohe = instrument_ohe
        if instrument_label is not None:
            self.current_instrument = instrument_label
        else:
            self.current_instrument = self.instruments[instrument_ohe.argmax()]

    def validate_current_track(self, track, instrument_ohe):
        """
        Validate that there is a current track initialized, if not, proceed to initialize it with the first
        element of the received parameters
        :param track: Tensor with the track's indexes
        :param instrument_ohe: The current instruments' OHE version
        :return:

        """
        if self.current_track is None or self.current_instrument is None or self.current_instrument_ohe is None:
            first_track = track[0]
            first_instrument_ohe = instrument_ohe[0]
            self.update_current_track(first_track, first_instrument_ohe)

    def instrument_ohe_to_name(self, instrument_ohe):
        return self.instruments[instrument_ohe.argmax()]

    def append_results_to_current_track(self, output):
        self.current_output[self.current_instrument].append(output)

    def add_new_predictions(self, output, track, instrument_ohe):
        # First, validate that the current instrument is initialized.
        self.validate_current_track(track, instrument_ohe)
        # If the received output has a different track and instrument when compared to the last track,
        # this means that the previous track has finished and we should calculate its results with museval
        same_instrument_mask = (instrument_ohe == self.current_instrument_ohe).all(dim=1)
        same_track_mask = (track == self.current_track)
        if ((~same_instrument_mask).any() or (~same_track_mask).any()) and self.get_current_output_length() == 0:
            raise Exception('Current track is empty, but received a different(new) instrument/track.')
        elif (~same_track_mask).any():
            # Get the observations that belong to the current instrument (and track) and append them to the results
            if same_track_mask.any():
                self.append_results_to_current_track(output[same_track_mask])
            # Retrieve the original track from musdb, generate the estimates from the model's output
            # and Evaluate the finished track
            mus_track = self.mus_db.tracks[self.current_track]
            estimates = {instru: torch.cat(output_list).transpose(0, 1).flatten(1, 2).cpu().numpy()[:,
                                 :len(mus_track)].T for instru, output_list in
                         self.current_output.items()}
            self.results.add_track(museval.eval_mus_track(mus_track, estimates))
            # Reset current results
            self.reset_current_results()
            self.validate_current_track(track[~same_track_mask], instrument_ohe[~same_track_mask])
            # Get the observations that belong to the new instrument and append them to the results
            self.append_results_to_current_track(output[~same_track_mask])
        elif (~same_instrument_mask).any():
            # Get the observations that belong to the current instrument and append them to the results
            if same_instrument_mask.any():
                self.append_results_to_current_track(output[same_instrument_mask])

            # Update current instrument and
            # Get the observations that belong to the new instrument and append them to the new results.
            self.update_current_instrument(instrument_ohe[~same_instrument_mask][0])
            self.append_results_to_current_track(output[~same_instrument_mask])

        # Add the received outputs into the current_results dictionary if there are no changes in instrument
        # or
        else:
            self.append_results_to_current_track(output)





