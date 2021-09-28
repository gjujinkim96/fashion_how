import os

import torch

from model import OneModel3


def get_model(setting, data_provider):
    if setting.model_name == 'OneModel3':
        model_class = OneModel3
    else:
        raise NotImplementedError()

    return model_class(setting, data_provider)


def load_model_from_saved(model, device, setting):
    if setting.model_file is not None:
        file_name = os.path.join(setting.model_save_dir, setting.model_file)
        if os.path.exists(file_name):
            checkpoint = torch.load(file_name, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model'])
            model = model.to(device)
            print('[*] load success: {}\n'.format(file_name))
            return model
        else:
            print('[!] checkpoints path does not exist...\n')
            return None
    else:
        return None
