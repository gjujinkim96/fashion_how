import os
import time
import timeit

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from tqdm import tqdm

from log_wandb import Logger, log_init, log_summary
from metrics import calculate_weighted_kendall_tau
from model_provider import load_model_from_saved


def output_from_batch(model, batch, criterion=None, is_train=True):
    rnk = batch[-1]
    batch = batch[:-1]
    logits = model(*batch)

    if criterion is not None:
        if is_train and model.multi_sample > 1:
            rnk = rnk.repeat(model.multi_sample)
        loss = criterion(logits, rnk)
        return logits, rnk, loss
    else:
        return logits, rnk


def pred_from_output(logits, is_train=True):
    logits = logits.detach().cpu()
    pred = torch.argmax(logits, 1)
    return pred


def run_epoch(setting, cur_epoch, model, dl, device,
              criterion=None, optimizer=None, lr_scheduler=None, scaler=None,
              is_test=False, no_log=False, swa_model=None, no_print=False):
    is_train = optimizer is not None
    use_amp = setting.use_amp and device == torch.device('cuda')
    return_loss = criterion is not None

    has_lr_scheduler = lr_scheduler is not None

    should_log = setting.should_log and cur_epoch >= 0
    logger = Logger(should_log)
    logger_prefix = 'train' if is_train else 'val'

    if model is None:
        swa_model.train(is_train)
    else:
        model.train(is_train)
    torch.set_grad_enabled(is_train)

    time_start = timeit.default_timer()
    losses = []
    preds = []
    rnks = []
    taus = []
    total_batch = len(dl)

    if not no_print:
        iter_bar = tqdm(enumerate(dl), total=total_batch)
        iter_bar.set_description(logger_prefix)
    else:
        iter_bar = enumerate(dl)
    for batch_idx, batch in iter_bar:
        if is_train:
            optimizer.zero_grad()

        batch = [t.to(device) for t in batch]

        with autocast(enabled=use_amp):
            if model is None:
                if return_loss:
                    output, rnk, loss = output_from_batch(swa_model, batch, criterion, is_train=is_train)
                else:
                    output, rnk = output_from_batch(swa_model, batch, is_train=is_train)
            else:
                if return_loss:
                    output, rnk, loss = model.output_from_batch(batch, criterion, is_train=is_train)
                else:
                    output, rnk = model.output_from_batch(batch, is_train=is_train)

        if is_train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), setting.max_grad_norm)
            scaler.step(optimizer)

            if swa_model is not None:
                swa_model.update_parameters(model)

            old_scale = scaler.get_scale()
            scaler.update()
            new_scale = scaler.get_scale()

            if has_lr_scheduler and old_scale == new_scale:
                lr_scheduler.step()

        if model is None:
            pred = pred_from_output(output, is_train=is_train)
        else:
            pred = model.pred_from_output(output, is_train=is_train)
        rnk = rnk.detach().cpu()

        if is_train:
            tau = calculate_weighted_kendall_tau(pred, rnk, setting.num_rnk)
            taus.append(tau)

            logger.reset(cur_epoch + batch_idx / total_batch)
            logger.add_log_item('loss', loss.detach().item(), logger_prefix)
            logger.add_log_item('tau', tau, logger_prefix)

            logger.add_log_item('lr', lr_scheduler.get_last_lr()[0])
            logger.log()

        preds.append(pred)
        if return_loss:
            losses.append(loss.item())
        rnks.append(rnk)

    time_spent = timeit.default_timer() - time_start
    mean_loss = torch.mean(torch.tensor(losses))
    preds = torch.cat(preds)
    rnks = torch.cat(rnks)

    if is_test:
        taus = calculate_weighted_kendall_tau(preds, rnks, setting.num_rnk)
        if not no_log:
            logger.reset(cur_epoch + 1)
            logger.add_log_item('loss', mean_loss, logger_prefix)
            logger.add_log_item('tau', taus, logger_prefix)
            logger.log()

    return mean_loss, preds, taus, time_spent


def run_train(setting, model, device, criterion, optimizer, lr_scheduler, train_dl, val_dl=None):
    log_init(setting)

    if not os.path.exists(setting.model_save_dir):
        os.makedirs(setting.model_save_dir)

    print('\n<Train>')
    print('total examples in train dataset: {}'.format(len(train_dl.dataset)))

    if val_dl is not None:
        print('\n<Evaluate>')
        print('total examples in valid dataset: {}'.format(len(val_dl.dataset)))
        print()

    use_amp = setting.use_amp and device == torch.device('cuda')
    scaler = GradScaler(enabled=use_amp)
    use_swa = setting.use_swa
    if use_swa:
        swa_model = AveragedModel(model)
        swa_start = setting.swa_start
        swa_scheduelr = SWALR(optimizer, anneal_strategy="linear", anneal_epochs=3, swa_lr=setting.swa_lr)
    else:
        swa_model = None

    best_val_taus = -2
    for epoch in range(setting.epochs):
        if use_swa and epoch > swa_start:
            lr_scheduler = swa_scheduelr

        mean_train_loss, _, _, train_time_spent = run_epoch(setting, epoch, model, train_dl, device,
                                                            criterion, optimizer, lr_scheduler, scaler,
                                                            swa_model=swa_model)

        print('-' * 50)
        print('Epoch: {}/{}'.format(epoch + 1, setting.epochs))
        print(f'Training Time: {time.strftime("%H:%M:%S", time.gmtime(train_time_spent))}')
        print('Loss: {:.4f}'.format(mean_train_loss))
        print('-' * 50)
        print()

        if val_dl is not None:
            mean_val_loss, _, val_taus, val_time_spent = run_epoch(setting, epoch, model, val_dl, device, criterion,
                                                                   is_test=True, swa_model=swa_model)

            if val_taus > best_val_taus:
                best_val_taus = val_taus
                log_summary('best_val_tau', best_val_taus)

                if setting.save_best:
                    file_name = os.path.join(setting.model_save_dir, f'{setting.model_name}-best.pt')
                    torch.save({'model': model.state_dict()}, file_name)

            print('-' * 50)
            print('Epoch: {}/{}'.format(epoch + 1, setting.epochs))
            print(f'Testing Time: {time.strftime("%H:%M:%S", time.gmtime(val_time_spent))}')
            print('Loss: {:.4f}'.format(mean_val_loss))
            print('-' * 50)
            print()

        if (epoch+1) % setting.save_freq == 0:
            file_name = os.path.join(setting.model_save_dir, f'{setting.model_name}-{epoch+1}.pt')
            torch.save({'model': model.state_dict()}, file_name)

    if use_swa:
        update_bn(train_dl, swa_model)
        best_val_taus = -2
        if val_dl is not None:
            mean_val_loss, _, val_taus, val_time_spent = run_epoch(setting, setting.epochs, None, val_dl, device,
                                                                   criterion, is_test=True, swa_model=swa_model)

            if val_taus > best_val_taus:
                best_val_taus = val_taus
                log_summary('best_val_tau', best_val_taus)

                if setting.save_best:
                    file_name = os.path.join(setting.model_save_dir, f'{setting.model_name}-best.pt')
                    torch.save({'model': model.state_dict()}, file_name)

            print('-' * 50)
            print('Epoch: {}/{}'.format(epoch + 1, setting.epochs))
            print(f'Testing Time: {time.strftime("%H:%M:%S", time.gmtime(val_time_spent))}')
            print('Loss: {:.4f}'.format(mean_val_loss))
            print(f'Tau: {val_taus:.4f}')
            print('-' * 50)
            print()

        file_name = os.path.join(setting.model_save_dir, f'{setting.model_name}-{epoch}.pt')
        torch.save({'model': model.state_dict()}, file_name)


def run_test(setting, model, test_dl, device, criterion):
    print('\n<Evaluate>')

    model = load_model_from_saved(model, device, setting)
    if model is None:
        return False


    _, preds, taus, test_time_spent = run_epoch(setting, -1, model, test_dl, device, is_test=True, no_log=True)

    print(f'Prediction Time: {time.strftime("%H:%M:%S", time.gmtime(test_time_spent))}')
    print('# of Test Examples: {}'.format(len(test_dl.dataset)))
    print(f'Tau: {taus:.4f}')
    print('-' * 50)


def run_pred(setting, model, pred_dl, device):
    print('\n<Predict>')

    model = load_model_from_saved(model, device, setting)
    if model is None:
        return False

    _, preds, _, pred_time_spent = run_epoch(setting, -1, model, pred_dl, device)

    # 실제 제출결과 생성시 경로는 '/home/work/model/prediction.csv'로 고정
    preds = preds.numpy()
    np.savetxt(setting.pred_csv_save_name, preds.astype(int), encoding='utf8', fmt='%d')
    print(f'saved result to {setting.pred_csv_save_name}')
    time_end = timeit.default_timer()

    print('-' * 50)
    print(f'Prediction Time: {time.strftime("%H:%M:%S", time.gmtime(pred_time_spent))}')
    print('# of Pred Examples: {}'.format(len(pred_dl.dataset)))
    print('-' * 50)
