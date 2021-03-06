import torch
import pathlib
import datetime
from trainer.evaluator import Evaluator
from utils.functions import get_dataloader_from_dataset
import os


class Trainer:
    def __init__(self, model, spectrogramer, optimizer, loss_function, lr_scheduler, train_dataset,
                 validation_dataset, log, checkpoint_folder_path, epochs, logging_frequency, checkpoint_frequency,
                 batch_size, prefetch_factor, instruments, model_configuration_name, device, checkpoint_filename,
                 evaluate_during_training):
        self.model = model.to(device)
        self.spectrogramer = spectrogramer
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.lr_scheduler = lr_scheduler
        self.log = log
        self.checkpoint_folder_path = checkpoint_folder_path
        self.epochs = epochs
        self.logging_frequency = logging_frequency
        self.checkpoint_frequency = checkpoint_frequency
        self.losses = []
        self.instruments = instruments
        self.model_configuration_name = model_configuration_name
        self.device = device
        self.checkpoint_filename = checkpoint_filename
        self.evaluate_during_training = evaluate_during_training
        self.start_epoch = 0
        self.train_loader = get_dataloader_from_dataset(train_dataset, 'train', device, batch_size)
        self.validation_loader = get_dataloader_from_dataset(validation_dataset, 'valid', device, batch_size)

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.evaluator = Evaluator(self.model,
                                   self.spectrogramer,
                                   loss_function,
                                   self.validation_loader,
                                   self.validation_dataset,
                                   self.instruments,
                                   self.device)
        self.load_model_checkpoint()

    def train(self, verbose=False):
        self.log.info("------------Training Started------------")
        for epoch in range(self.start_epoch, self.epochs):  # loop over the dataset multiple times
            self.log.info("!!------------ Epoch {} ------------!!".format(epoch))
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                # Here the inputs are the mixture track, targets are the final track and condition is the OHE instrument
                inputs, target, condition = data
                # Send to the correct device
                inputs, target, condition = inputs.to(self.device), target.to(self.device), condition.to(self.device)
                # (bs, time, ch), (bs, time, ch),
                if verbose: self.log.info('--- Mini-Batch {} - Condition shape {}'.format(i, str(list(condition.shape))))
                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Get spectrograms, dimensions: (bs, ch, n_fft, time)
                inputs_spec, inp_phase = self.spectrogramer.get_spectrogram(inputs)
                target_spec, targ_phase = self.spectrogramer.get_spectrogram(target)

                # audio = self.spectrogramer.reconstruct(target_spec, targ_phase)
                # The output is a soft mask that we will use for 'filtering' the input spectrogram
                outputs = self.model(inputs_spec, condition)
                outputs = outputs * inputs_spec
                loss = self.loss_function(target_spec, outputs)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                self.losses.append(loss.item())
                if i % self.logging_frequency == self.logging_frequency - 1:
                    self.log.info('[Epoch %d, Iter. %5d] avg. loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / self.logging_frequency))
                    running_loss = 0.0

            if (epoch % self.checkpoint_frequency) == (self.checkpoint_frequency - 1):
                self.create_checkpoint(epoch + 1)
        if not (self.epochs % self.checkpoint_frequency) == 0:
            self.create_checkpoint(self.epochs, training_losses=self.losses)
        self.log.info('------------Training Finished------------')

    def create_checkpoint(self, epoch, training_losses=None):
        utc0_now_str = datetime.datetime.utcnow().strftime('%Y-%m-%d-%H%M')
        checkpoint_filepath = pathlib.Path(self.checkpoint_folder_path,
                                           "{}_{}_{}.pt".format(self.model_configuration_name, int(epoch),
                                                                utc0_now_str))
        self.log.info('Creating Checkpoint for Epoch {}. Checkpoint location: {}'.format(epoch, checkpoint_filepath))
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if training_losses is not None:
            checkpoint_dict['training_losses'] = training_losses
        if self.evaluate_during_training:
            val_loss, val_track_scores_df, val_scores_df = self.evaluator.evaluate()
            checkpoint_dict['val_loss']: val_loss
            checkpoint_dict['val_track_scores_df']: val_track_scores_df
            checkpoint_dict['val_scores_df']: val_scores_df
        # Saves checkpoint in a file with unique name, in case we want to access this specific version of the model
        torch.save(checkpoint_dict, checkpoint_filepath)
        # Saves checkpoint in the generic file that contains the latest checkpoint available.
        torch.save(checkpoint_dict, os.path.join(self.checkpoint_folder_path, self.checkpoint_filename))
        self.log.info('Checkpoint successful for Epoch {}. Checkpoint location: {}'.format(epoch, checkpoint_filepath))

    def get_loss_historic(self):
        return self.losses

    def get_results(self):
        return self.evaluator.musdb18evaluator.get_results_df()

    def load_model_checkpoint(self):
        """
        This function loads a model checkpoint if there is a checkpoint with the models configuration name
        :return:
        """
        checkpoint_abs_path = os.path.join(self.checkpoint_folder_path, self.checkpoint_filename)
        if os.path.exists(checkpoint_abs_path):
            checkpoint = torch.load(checkpoint_abs_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.log.info('Loaded {} model. Starting from the '
                          'epoch number {}. Checkpoint path: {}'.format(self.model_configuration_name,
                                                                        self.start_epoch, checkpoint_abs_path))
            self.model.train()
        else:
            self.log.info('No checkpoint found at {}. Training will start from scratch.'.format(checkpoint_abs_path))