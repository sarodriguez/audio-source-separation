import torch.optim
from torch import nn
from torch.utils.data import DataLoader


def get_optimizer_class(optimizer_name):
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adam':
        return torch.optim.Adam
    else:
        raise NotImplementedError


def get_activation(activation_name):
    if activation_name == 'relu':
        return nn.ReLU()
    elif activation_name == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation_name == 'sigmoid':
        return nn.Sigmoid()
    elif activation_name in ('identity', 'no_activation'):
        return nn.Identity()
    else:
        raise NotImplementedError


def get_lr_scheduler_class(lr_scheduler_name):
    lr_scheduler_name = lr_scheduler_name.lower()
    if lr_scheduler_name == 'cosine_annealing':
        return torch.optim.lr_scheduler.CosineAnnealingLR
    else:
        raise NotImplementedError


def get_loss_function(loss_function_name):
    loss_function_name = loss_function_name.lower()
    if loss_function_name in ('mean_absolute_error', 'l1loss'):
        return nn.L1Loss()
    elif loss_function_name == 'mean_squared_error':
        return nn.MSELoss()
    else:
        raise NotImplementedError


def get_device(gpu: bool, gpu_device: str):
    """
    Get a device given an indicator to use gpu or not, and a a gpu device
    :param gpu: Boolean value, True means use gpu false otherwise.
    :param gpu_device:
    :return:
    """
    if gpu and torch.cuda.is_available():
        if gpu_device is not None:
            device = gpu_device
        else:
            device = 'cuda'
    else:
        device = 'cpu'
    return device


def is_device_gpu(device: str):

    return 'cuda' in device


def get_stft_window(window_type, window_length):
    if window_type == 'hamming':
        return torch.nn.Parameter(torch.hann_window(window_length))


def get_dataloader_from_dataset(dataset, subset_split, device, batch_size):
    if subset_split == 'train':
        if is_device_gpu(device):
            # pin memory helps improving performance when loading on CPU and sending to GPU
            # The drop last flag is used to avoid problems with batch norm on a single observation batch at the end.
            data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, drop_last=True)
        else:
            data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=1, drop_last=True)
    else:
        if is_device_gpu(device):
            data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
        else:
            data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=1)
    return data_loader
