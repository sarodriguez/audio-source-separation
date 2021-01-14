import os
import yaml

from config import config
from utils.log import setup_log
from datasets.musdb18_dataset import TrainMUSDB18Dataset, TestMUSDB18Dataset
from trainer.evaluator import Evaluator
from models.cunet import ConditionedUNet
from utils.functions import get_optimizer_class, get_lr_scheduler_class, get_loss_function, get_device, get_dataloader_from_dataset
from utils.spectrogram import Spectrogramer

def main():
    # Setup the log
    log_filename = config.LOG_FILENAME

    log = setup_log(log_filename)
    log.info("Started Evaluation Setup")

    # Setup datasets
    instruments = config.INSTRUMENTS
    musdbwav_path = config.MUSDB18_WAV_PATH
    # sample_time_frames = config.SAMPLE_TIME_FRAMES
    audio_samples_per_chunk = config.AUDIO_SAMPLES_PER_CHUNK
    batch_size = config.BATCH_SIZE
    prefetch_factor = config.PREFETCH_FACTOR

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
    checkpoint_filename = config.CHECKPOINT_FILENAME
    loss_function = get_loss_function(config.LOSS_FUNCTION)
    gpu = config.GPU
    gpu_device = config.GPU_DEVICE
    device = get_device(gpu, gpu_device)


    # Setup the spectrogram
    spectrogram_type = config.SPECTROGRAM_TYPE
    n_fft = config.N_FFT
    hop_length = config.HOP_LENGTH
    window = config.WINDOW
    window_length = config.WINDOW_LENGTH

    spectrogramer = Spectrogramer(spectrogram_type, n_fft, hop_length, window, window_length, device)


    # Evaluation parameters
    evaluate_dataset = config.EVALUATE_DATASET

    # Initialize dataset and dataloader
    if evaluate_dataset in ('valid', 'test'):
        dataset = TestMUSDB18Dataset(musdbwav_path, instruments, sample_length=audio_samples_per_chunk,
                                     subset_split=evaluate_dataset)
        dataloader = get_dataloader_from_dataset(dataset,evaluate_dataset,device, batch_size)
    else:
        raise Exception('The dataset for evaluation should be either test or valid.')
    train_dataset = TrainMUSDB18Dataset(musdbwav_path, instruments, sample_length=audio_samples_per_chunk)

    log.info("Loaded Training dataset with a total of {} samples, from the path: {}".format(len(train_dataset.mus),
                                                                                            musdbwav_path))
    log.info("Loaded Validation dataset with a total of {} samples, from the path: {}".format(len(train_dataset.mus),
                                                                                              musdbwav_path))
    # Initialize evaluator
    evaluator = Evaluator(model,
                          spectrogramer,
                          loss_function,
                          dataloader,
                          dataset,
                          instruments,
                          device)
    evaluator.load_model_checkpoint(os.path.join(checkpoint_folder_path, checkpoint_filename))

    # Start trainer/evaluation
    log.info("Evaluation on the {} dataset split has started.")
    evaluator.evaluate()


if __name__ == '__main__':
    config.parse_args()
    main()