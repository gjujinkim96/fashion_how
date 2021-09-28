import transformers
import torch.optim as optim


def get_lr_scheduler(setting, optimizer, dataloader):
    if setting.lr_scheduler == 'get_cosine_with_hard_restarts_schedule_with_warmup':
        total_steps = setting.epochs * len(dataloader)

        if setting.warmup_steps is not None:
            warmup_steps = setting.warmup_steps
        else:
            warmup_steps = int(total_steps * setting.warmup_ratio)
        return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            warmup_steps,
            total_steps,
            setting.num_cycles,
        )
    elif setting.lr_scheduler == 'get_cosine_schedule_with_warmup':
        total_steps = setting.epochs * len(dataloader)

        if setting.warmup_steps is not None:
            warmup_steps = setting.warmup_steps
        else:
            warmup_steps = int(total_steps * setting.warmup_ratio)

        return transformers.get_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps,
            total_steps,
        )
    elif setting.lr_scheduler == 'get_constant_schedule':
        return transformers.get_constant_schedule(optimizer)
    elif setting.lr_scheduler == 'StepLR':
        steps = setting.step_size_n_epoch * len(dataloader)
        return optim.lr_scheduler.StepLR(optimizer, steps, setting.step_size_gamma)
    else:
        raise NotImplementedError()
