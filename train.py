import os
import yaml

from config import config
from utils.log import setup_log
from datasets.musdb18_dataset import TrainMUSDB18Dataset, TestMUSDB18Dataset
from trainer.trainer import Trainer
from models.cunet import ConditionedUNet
from utils.functions import get_optimizer_class, get_lr_scheduler_class, get_loss_function, get_device
from utils.spectrogram import Spectrogramer


def main():
    # Setup the log
    log_filename = config.LOG_FILENAME

    log = setup_log(log_filename)
    log.info("Started Training")

    # Setup datasets
    instruments = config.INSTRUMENTS
    musdbwav_path = config.MUSDB18_WAV_PATH
    # sample_time_frames = config.SAMPLE_TIME_FRAMES
    audio_samples_per_chunk = config.AUDIO_SAMPLES_PER_CHUNK
    batch_size = config.BATCH_SIZE
    prefetch_factor = config.PREFETCH_FACTOR

    train_dataset = TrainMUSDB18Dataset(musdbwav_path, instruments, sample_length=audio_samples_per_chunk)
    validation_dataset = TestMUSDB18Dataset(musdbwav_path, instruments, sample_length=audio_samples_per_chunk,
                                            subset_split='valid')
    log.info("Loaded Training dataset with a total of {} samples, from the path: {}".format(len(train_dataset.mus),
                                                                                            musdbwav_path))
    log.info(
        "Loaded Validation dataset with a total of {} samples, from the path: {}".format(len(validation_dataset.mus),
                                                                                         musdbwav_path))

    # Setup model
    model_config_path = config.MODEL_CONFIG_YAML_PATH
    model_config_name = config.MODEL_CONFIGURATION

    with open(model_config_path, 'r') as file:
        model_configurations = yaml.load(file, Loader=yaml.FullLoader)
    train_model_config = model_configurations[model_config_name]
    train_model_class = train_model_config.pop('class')
    model = eval(train_model_class)(**train_model_config)

    # Setup trainer
    checkpoint_folder_path = config.CHECKPOINT_FOLDER_PATH
    epochs = config.EPOCHS
    checkpoint_frequency = config.CHECKPOINT_FREQUENCY
    checkpoint_filename = config.CHECKPOINT_FILENAME
    logging_frequency = config.LOGGING_FREQUENCY
    optimizer_class = get_optimizer_class(config.OPTIMIZER)
    optimizer_params = config.OPTIMIZER_PARAMS
    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    lr_scheduler_class = get_lr_scheduler_class(config.LR_SCHEDULER)
    lr_scheduler_params = config.LR_SCHEDULER_PARAMS
    lr_scheduler = lr_scheduler_class(optimizer, **lr_scheduler_params)
    loss_function = get_loss_function(config.LOSS_FUNCTION)
    gpu = config.GPU
    gpu_device = config.GPU_DEVICE
    device = get_device(gpu, gpu_device)
    evaluate_during_training = config.EVALUATE_DURING_TRAINING

    if not os.path.isdir(checkpoint_folder_path):
        os.mkdir(checkpoint_folder_path)

    # Setup the spectrogram
    spectrogram_type = config.SPECTROGRAM_TYPE
    n_fft = config.N_FFT
    hop_length = config.HOP_LENGTH
    window = config.WINDOW
    window_length = config.WINDOW_LENGTH

    spectrogramer = Spectrogramer(spectrogram_type, n_fft, hop_length, window, window_length, device)

    # Initialize trainer
    trainer = Trainer(model, spectrogramer, optimizer, loss_function, lr_scheduler, train_dataset,
                      validation_dataset, log, checkpoint_folder_path, epochs, logging_frequency, checkpoint_frequency,
                      batch_size, prefetch_factor, instruments, train_model_class, device, checkpoint_filename,
                      evaluate_during_training)

    # Start trainer/evaluation
    trainer.train()


if __name__ == '__main__':
    config.parse_args()
    main()
