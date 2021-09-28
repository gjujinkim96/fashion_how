import os
import random

import numpy as np
import torch


def set_random_seed(setting):
    torch.manual_seed(setting.seed)
    random.seed(setting.seed)
    np.random.seed(setting.seed)

    try:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_torch_thread():
    cores = os.cpu_count()
    torch.set_num_threads(cores)


def get_udevice(setting):
    """
    function: get usable devices(CPU and GPU)
    """
    if torch.cuda.is_available() and setting.cuda_pref:
        device = torch.device('cuda')
        num_gpu = torch.cuda.device_count()
    else:
        device = torch.device('cpu')
    print('Using device: {}'.format(device))
    if device == torch.device('cuda'):
        print('# of GPU: {}'.format(num_gpu))
    return device
