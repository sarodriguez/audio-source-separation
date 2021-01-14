import effortless_config


class config(effortless_config.Config):

    """
    log parameters
    """
    LOG_FILENAME = r'C:\Proy\sound-source-separation\logscheckpoints\log.log'
    """
    spectrogram parameters
    """
    # since we are using musdb, we do not actually reference sample rate anywhere
    # sample_rate = 44100
    # Either complex or magnitude, only magnitude supported so far
    SPECTROGRAM_TYPE = 'magnitude'
    N_FFT = 1024
    HOP_LENGTH = 256
    WINDOW = 'hamming'
    WINDOW_LENGTH = N_FFT
    # this parameter controls the length of the clips to be used and, thus, directly relates to the model parameters
    # We are going to use SAMPLE_TIME_FRAMES as the input of the models
    SAMPLE_TIME_FRAMES = 128
    # Where the length of each time frame equals the hop length
    TIME_FRAME_LENGTH = HOP_LENGTH
    AUDIO_SAMPLES_PER_CHUNK = (SAMPLE_TIME_FRAMES - 1) * TIME_FRAME_LENGTH
    """
    dataset parameters
    """
    MUSDB18_WAV_PATH = r'c:\proy\sound-source-separation\data\musdb18wav'
    TRACKS = ['bass', 'drums', 'other', 'vocals', 'mix']
    INSTRUMENTS = ['bass', 'drums', 'other', 'vocals']

    """
    training parameters
    """
    GPU = True
    # Specifies a device to be used if not none.
    # Make sure that GPU_DEVICE can be sent as a parameter to torch's to function - .to(GPU_DEVICE)
    GPU_DEVICE = None
    BATCH_SIZE = 2
    PREFETCH_FACTOR = None
    EPOCHS = 200
    OPTIMIZER = 'adam'
    OPTIMIZER_PARAMS = dict(lr=1e-3)
    LOSS_FUNCTION = 'mean_absolute_error'
    LR_SCHEDULER = 'cosine_annealing'
    LR_SCHEDULER_PARAMS = dict(T_max=EPOCHS)
    CHECKPOINT_FOLDER_PATH = r'C:\Proy\sound-source-separation\logscheckpoints'
    # Continue training from an exisiting checkpoint, if the file does not exist, training will start from scratch
    # The last checkpoint will be stored in the same path, historical checkpoints will be saved with a different
    # name that includes the model configuration name and the current timestamp.
    CHECKPOINT_FILENAME = 'ConditionedUNet.pt'
    # How many batch iterations between one log message of the current Loss
    LOGGING_FREQUENCY = 3000
    # How many epochs between one network backup and another
    CHECKPOINT_FREQUENCY = 10
    # This determines if the model should be evaluated during training or not. If true, the evaluation results df
    # will be stored with every created checkpoint
    EVALUATE_DURING_TRAINING = False
    """
    model parameters    
    """
    MODEL_CONFIG_YAML_PATH = 'model_config.yaml'
    # this is the model configuration that is going to be trained,
    MODEL_CONFIGURATION = 'default'

    """
    Evaluation parameters, only used in the evaluate script.
    """
    # This parameter indicates which dataset to use for evaluation, either test or validation.
    EVALUATE_DATASET = effortless_config.setting(default='test', validation='valid', test='test')

