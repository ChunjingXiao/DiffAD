import argparse
import logging
import math

import torch
from tensorboardX import SummaryWriter

import core.logger as Logger
import data as Data
import model as Model

# train model
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/smap_time_train.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)

    logger.info('Initial Dataset Finished')

    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_epoch = opt['train']['n_epoch']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    save_model_iter = math.ceil(train_set.__len__() / opt['datasets']['train']['batch_size'])
    while current_epoch < n_epoch:
        current_epoch += 1
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_epoch > n_epoch:
                break
            diffusion.feed_data(train_data)
            diffusion.optimize_parameters()
            # log
            if current_epoch % opt['train']['print_freq'] == 0 and current_step % save_model_iter == 0:
                logs = diffusion.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                    current_epoch, current_step)
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    tb_logger.add_scalar(k, v, current_step)
                logger.info(message)

            # save model
            if current_epoch % opt['train']['save_checkpoint_freq'] == 0 and current_step % save_model_iter == 0:
                logger.info('Saving models and training states.')
                diffusion.save_network(current_epoch, current_step)

    logger.info('End of training.')
