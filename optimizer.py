import torch.optim as optim
from madgrad import MADGRAD


def get_optimizer(setting, model):
    if setting.optimizer == 'SGD':
        return optim.SGD(model.parameters(), lr=setting.learning_rate,
                         momentum=setting.momentum, weight_decay=setting.weight_decay)
    elif setting.optimizer == 'AdamW':
        return optim.AdamW(model.parameters(), lr=setting.learning_rate,
                           weight_decay=setting.weight_decay, amsgrad=setting.amsgrad)
    elif setting.optimizer == 'MADGRAD':
        return MADGRAD(model.parameters(), lr=setting.learning_rate,
                       momentum=setting.momentum, weight_decay=setting.weight_decay)
    else:
        raise NotImplementedError()
