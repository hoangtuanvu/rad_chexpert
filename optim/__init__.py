from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import numpy as np
from .adabound import AdaBound
from .RAdam import RAdam, AdamW, PlainRAdam
import torch.optim as optim


def create_optimizer(args, model_params):
    logging.debug('Training Optimizer {}'.format(args.optim))
    if args.optim == 'sgd':
        return optim.SGD(model_params, args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay, nesterov=True)
    elif args.optim == 'adam':
        return optim.Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay)
    elif args.optim == 'amsgrad':
        return optim.Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, amsgrad=True)
    elif args.optim == 'adamW':
        return AdamW(model_params, args.lr, betas=(args.beta1, args.beta2),
                     weight_decay=args.weight_decay)
    elif args.optim == 'radam':
        return RAdam(params=model_params, lr=args.lr, betas=(args.beta1, args.beta2),
                     weight_decay=args.weight_decay)
    elif args.optim == 'plain_radam':
        return PlainRAdam(params=model_params, lr=args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay)
    elif args.optim == 'adabound':
        return adabound.AdaBound(model_params, args.lr, betas=(args.beta1, args.beta2),
                                 weight_decay=args.weight_decay)
    else:
        assert args.optim == 'amsbound'
        return adabound.AdaBound(model_params, args.lr, betas=(args.beta1, args.beta2),
                                 weight_decay=args.weight_decay, amsbound=True)


def create_scheduler(args, start_epoch):
    """
        Learning rate schedule with respect to epoch
        lr: float, initial learning rate
        lr_factor: float, decreasing factor every epoch_lr
        epoch_now: int, the current epoch
        lr_epochs: list of int, decreasing every epoch in lr_epochs
        return: lr, float, scheduled learning rate.
        """
    count = 0
    for epoch in args.lr_epochs.split(','):
        if start_epoch >= int(epoch):
            count += 1
            continue

        break

    return args.lr * np.power(args.lr_factor, count)
