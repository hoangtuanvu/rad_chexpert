# Author: Brian

# -----------------------------------------------------------------------
import uuid
import copy
import tqdm
import logging
import os
import time
import json
import random
import argparse
import numpy as np
from easydict import EasyDict as edict
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from losses import init_loss_func
from optim import create_optimizer
from dataflow import init_dataset, init_labels
from models import Classifier
from metrics import get_metrics
from pandas import DataFrame
from models.utils import load_state_dict, AverageMeter


class MLCModel:
    """Summary
    Attributes:
        cfg (TYPE): Description
        criterion (TYPE): Description
        device (TYPE): cpu or gpu
        hparams (TYPE): hyper parameters from parser
        labels (TYPE): list of the diseases, see init_labels()
        model (TYPE): feature extraction backbone with classifier, see cfg.json
        names (TYPE): list of filenames in the images which have been from dataloader
        num_tasks (TYPE): 5 or 14, number of diseases
    """

    def __init__(self, hparams):
        """Summary

        Args:
            hparams (TYPE): hyper parameters from parser
        """
        super(MLCModel, self).__init__()
        self.hparams = hparams
        self.device = torch.device(
            "cuda:{}".format(hparams.gpus) if torch.cuda.is_available() else "cpu")

        with open(self.hparams.json_path, 'r') as f:
            self.cfg = edict(json.load(f))
            hparams_dict = vars(self.hparams)
            self.cfg['hparams'] = hparams_dict
            if self.hparams.verbose is True:
                print(json.dumps(self.cfg, indent=4))

        if self.cfg.criterion in ['bce', 'focal', 'sce', 'bce_v2', 'bfocal']:
            self.criterion = init_loss_func(self.cfg.criterion, device=self.device)
        elif self.cfg.criterion == 'class_balance':
            samples_per_cls = list(map(int, self.cfg.samples_per_cls.split(',')))
            self.criterion = init_loss_func(self.cfg.criterion, samples_per_cls=samples_per_cls,
                                            loss_type=self.cfg.loss_type)
        else:
            self.criterion = init_loss_func(self.cfg.criterion)

        self.labels = init_labels(name=self.hparams.data_name)
        if self.cfg.extract_fields is None:
            self.cfg.extract_fields = ','.join([str(idx) for idx in range(len(self.labels))])
        else:
            assert isinstance(self.cfg.extract_fields, str), "extract_fields must be string!"

        self.model = Classifier(self.cfg, self.hparams)
        self.state_dict = None
        # Load cross-model from other configuration
        if self.hparams.load is not None and len(self.hparams.load) > 0:
            if not os.path.exists(hparams.load):
                raise ValueError('{} does not exists!'.format(hparams.load))
            state_dict = load_state_dict(self.hparams.load, self.model, self.device)
            self.state_dict = state_dict

        # DataParallel model
        if torch.cuda.device_count() > 1 and self.hparams.gpus == 0:
            self.model = nn.DataParallel(self.model)

        self.model.to(device=self.device)
        self.num_tasks = list(map(int, self.cfg.extract_fields.split(',')))
        self.names = list()
        self.optimizer, self.scheduler = self.configure_optimizers()
        self.train_loader = self.train_dataloader()
        self.valid_loader = self.val_dataloader()
        self.test_loader = self.test_dataloader()

    def forward(self, x):
        """Summary

        Args:
            x (TYPE): image

        Returns:
            TYPE: Description
        """
        return self.model(x)

    def train(self):
        epoch_start = 0

        summary_train = {'epoch': 0, 'step': 0, 'total_step': len(self.train_loader)}
        summary_dev = {'loss': float('inf'), 'score': 0.0}
        best_dict = {"score_dev_best": 0.0, "loss_dev_best": float('inf'),
                     "score_top_k": [0.0], "loss_top_k": [0.0],
                     "score_curr_idx": 0, "loss_curr_idx": 0}

        if self.state_dict is not None:
            summary_train = {'epoch': self.state_dict['epoch'], 'step': self.state_dict['step'],
                             'total_step': len(self.train_loader)}
            best_dict['score_dev_best'] = self.state_dict['score_dev_best']
            best_dict['loss_dev_best'] = self.state_dict['loss_dev_best']
            epoch_start = self.state_dict['epoch']

        for epoch in range(epoch_start, self.hparams.epochs):
            lr = self.create_scheduler(start_epoch=summary_train['epoch'])
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            logging.info('Learning rate in epoch {}: {}'.format(epoch + 1, self.optimizer.
                                                                param_groups[0]['lr']))
            print('Learning rate in epoch {}: {}'.format(epoch + 1,
                                                         self.optimizer.param_groups[0]['lr']))

            summary_train, best_dict = self.training_step(summary_train, summary_dev, best_dict)

            self.validation_end(summary_dev, summary_train, best_dict)

            torch.save({'epoch': summary_train['epoch'],
                        'step': summary_train['step'],
                        'score_dev_best': best_dict['score_dev_best'],
                        'loss_dev_best': best_dict['loss_dev_best'],
                        'state_dict': self.model.state_dict()},
                       os.path.join(self.hparams.save_path, '{}_model.pth'.
                                    format(summary_train['epoch'] - 1)))

        logging.info('Training finished, model saved')
        print('Training finished, model saved')

    # def training_step(self, batch, batch_nb):
    def training_step(self, summary_train, summary_dev, best_dict):
        """Summary
        Extract the batch of datapoints and return the predicted logits
        Args:
            summary_train:
            summary_dev:
            best_dict:
        Returns:
            TYPE: Description
        """
        losses = AverageMeter()
        torch.set_grad_enabled(True)
        self.model.train()
        time_now = time.time()

        for i, (inputs, target, _) in enumerate(self.train_loader):
            if isinstance(inputs, tuple):
                inputs = tuple(
                    [e.to(self.device) if type(e) == torch.Tensor else e for e in inputs])
            else:
                inputs = inputs.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()

            if self.cfg.no_jsd:
                if self.cfg.n_crops:
                    bs, n_crops, c, h, w = inputs.size()
                    inputs = inputs.view(-1, c, h, w)

                    if len(self.hparams.mixtype) > 0:
                        if self.hparams.multi_cls:
                            target = target.view(target.size()[0], -1)
                            inputs, targets_a, targets_b, lam = self.mix_data(inputs,
                                                                              target.repeat(1,
                                                                                            n_crops).view(
                                                                                  -1), self.device,
                                                                              self.hparams.alpha)
                        else:
                            inputs, targets_a, targets_b, lam = self.mix_data(inputs,
                                                                              target.repeat(1,
                                                                                            n_crops).view(
                                                                                  -1, len(
                                                                                      self.num_tasks)),
                                                                              self.device,
                                                                              self.hparams.alpha)

                    logits = self.forward(inputs)
                    if len(self.hparams.mixtype) > 0:
                        loss_func = self.mixup_criterion(targets_a, targets_b, lam)
                        loss = loss_func(self.criterion, logits)
                    else:
                        if self.hparams.multi_cls:
                            target = target.view(target.size()[0], -1)
                            loss = self.criterion(logits, target.repeat(1, n_crops).view(-1))
                        else:
                            loss = self.criterion(logits, target.repeat(1, n_crops).view(-1, len(
                                self.num_tasks)))
                else:
                    if len(self.hparams.mixtype) > 0:
                        inputs, targets_a, targets_b, lam = self.mix_data(inputs, target,
                                                                          self.device,
                                                                          self.hparams.alpha)

                    logits = self.forward(inputs)
                    if len(self.hparams.mixtype) > 0:
                        loss_func = self.mixup_criterion(targets_a, targets_b, lam)
                        loss = loss_func(self.criterion, logits)
                    else:
                        loss = self.criterion(logits, target)
            else:
                images_all = torch.cat(inputs, 0)
                logits_all = self.forward(images_all)
                logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, inputs[0].size(0))

                # Cross-entropy is only computed on clean images
                loss = F.cross_entropy(logits_clean, target)

                p_clean, p_aug1, p_aug2 = F.softmax(logits_clean, dim=1), F.softmax(logits_aug1,
                                                                                    dim=1), F.softmax(
                    logits_aug2, dim=1)

                # Clamp mixture distribution to avoid exploding KL divergence
                p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
                loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') + F.kl_div(
                    p_mixture, p_aug1, reduction='batchmean') + F.kl_div(p_mixture, p_aug2,
                                                                         reduction='batchmean')) / 3.

            assert not np.isnan(loss.item()), 'Model diverged with losses = NaN'

            loss.backward()
            self.optimizer.step()
            summary_train['step'] += 1
            losses.update(loss.item(), target.size(0))

            if summary_train['step'] % self.hparams.log_every == 0:
                time_spent = time.time() - time_now
                time_now = time.time()
                logging.info('Train, '
                             'Epoch : {}, '
                             'Step : {}/{}, '
                             'Loss: {loss.val:.4f} ({loss.avg:.4f}), '
                             'Run Time : {runtime:.2f} sec'.format(summary_train['epoch'] + 1,
                                                                   summary_train['step'],
                                                                   summary_train['total_step'],
                                                                   loss=losses, runtime=time_spent))
                print('Train, '
                             'Epoch : {}, '
                             'Step : {}/{}, '
                             'Loss: {loss.val:.4f} ({loss.avg:.4f}), '
                             'Run Time : {runtime:.2f} sec'.format(summary_train['epoch'] + 1,
                                                                   summary_train['step'],
                                                                   summary_train['total_step'],
                                                                   loss=losses, runtime=time_spent))

            if summary_train['step'] % self.hparams.test_every == 0:
                self.validation_end(summary_dev, summary_train, best_dict)

            self.model.train()
            torch.set_grad_enabled(True)

        summary_train['epoch'] += 1
        return summary_train, best_dict

    def validation_step(self, summary_dev):
        """Summary
        Extract the batch of datapoints and return the predicted logits in validation step
        Args:
            summary_dev (TYPE): Description

        Returns:
            TYPE: Description
        """
        losses = AverageMeter()
        torch.set_grad_enabled(False)
        self.model.eval()

        output_ = np.array([])
        target_ = np.array([])

        with torch.no_grad():
            for i, (inputs, target, _) in enumerate(self.valid_loader):
                target = target.to(self.device)
                if isinstance(inputs, tuple):
                    inputs = tuple(
                        [e.to(self.device) if type(e) == torch.Tensor else e for e in inputs])
                else:
                    inputs = inputs.to(self.device)

                logits = self.forward(inputs)
                loss = self.criterion(logits, target)
                losses.update(loss.item(), target.size(0))

                if self.hparams.multi_cls:
                    output = F.softmax(logits)
                    _, output = torch.max(output, 1)
                else:
                    output = torch.sigmoid(logits)

                target = target.detach().to('cpu').numpy()
                target_ = np.concatenate((target_, target), axis=0) if len(target_) > 0 else target
                y_pred = output.detach().to('cpu').numpy()
                output_ = np.concatenate((output_, y_pred), axis=0) if len(output_) > 0 else y_pred

        summary_dev['loss'] = losses.avg
        return summary_dev, output_, target_

    def validation_end(self, summary_dev, summary_train, best_dict):
        """Summary
        After the validation end, calculate the metrics
        Args:
            summary_dev (TYPE): Description
            summary_train (TYPE): Description
            best_dict (TYPE): Description

        Returns:
            TYPE: Description
        """
        time_now = time.time()
        summary_dev, output_, target_ = self.validation_step(summary_dev)
        time_spent = time.time() - time_now

        if not self.hparams.auto_threshold:
            overall_pre, overall_rec, overall_fscore = get_metrics(copy.deepcopy(output_), target_,
                                                                   self.cfg.beta,
                                                                   self.cfg.threshold,
                                                                   self.cfg.metric_type)
        else:
            overall_pre, overall_rec, overall_fscore = self.find_best_fixed_threshold(output_,
                                                                                      target_)

        resp = dict()
        if not self.hparams.multi_cls:
            for t in range(len(self.num_tasks)):
                y_pred = np.transpose(output_)[t]
                precision, recall, f_score = get_metrics(copy.deepcopy(y_pred),
                                                         np.transpose(target_)[t], self.cfg.beta,
                                                         self.cfg.threshold, 'binary')

                resp['precision_{}'.format(self.labels[self.num_tasks[t]])] = precision
                resp['recall_{}'.format(self.labels[self.num_tasks[t]])] = recall
                resp['f_score_{}'.format(self.labels[self.num_tasks[t]])] = f_score

        resp['overall_precision'] = overall_pre
        resp['overall_recall'] = overall_rec
        resp['overall_f_score'] = overall_fscore

        logging.info('Dev, Step : {}/{}, Loss : {}, Fscore : {:.3f}, Precision : {:.3f}, '
                     'Recall : {:.3f}, Run Time : {:.2f} sec'.format(
                        summary_train['step'],
                        summary_train['total_step'],
                        summary_dev['loss'],
                        resp['overall_f_score'],
                        resp['overall_precision'],
                        resp['overall_recall'], time_spent))
        print('Dev, Step : {}/{}, Loss : {}, Fscore : {:.3f}, Precision : {:.3f}, '
                     'Recall : {:.3f}, Run Time : {:.2f} sec'.format(
                        summary_train['step'],
                        summary_train['total_step'],
                        summary_dev['loss'],
                        resp['overall_f_score'],
                        resp['overall_precision'],
                        resp['overall_recall'], time_spent))

        save_best = False
        mean_score = resp['overall_f_score']
        if mean_score > min(best_dict['score_top_k']):
            self.update_top_k(mean_score, best_dict, 'score')
            if self.hparams.metric == 'score':
                save_best = True

        mean_loss = summary_dev['loss']
        if mean_loss < max(best_dict['loss_top_k']):
            self.update_top_k(mean_loss, best_dict, 'loss')
            if self.hparams.metric == 'loss':
                save_best = True

        if save_best:
            torch.save({'epoch': summary_train['epoch'], 'step': summary_train['step'],
                        'score_dev_best': best_dict['score_dev_best'],
                        'loss_dev_best': best_dict['loss_dev_best'],
                        'state_dict': self.model.state_dict()},
                       os.path.join(self.hparams.save_path, 'best{}.pth'.
                                    format(best_dict['score_curr_idx'])))

            logging.info(
                'Best {}, Step : {}/{}, Loss : {}, Score : {:.3f}'.format(
                    best_dict['score_curr_idx'],
                    summary_train['step'],
                    summary_train['total_step'],
                    summary_dev['loss'],
                    best_dict['score_dev_best']))

            print('Best {}, Step : {}/{}, Loss : {}, Score : {:.3f}'.format(
                    best_dict['score_curr_idx'],
                    summary_train['step'],
                    summary_train['total_step'],
                    summary_dev['loss'],
                    best_dict['score_dev_best']))

    def find_best_fixed_threshold(self, output_, target_):
        score = list()
        thrs = np.arange(0, 1.0, 0.01)
        pre_rec = list()
        for thr in tqdm.tqdm(thrs):
            pre, rec, fscore = get_metrics(copy.deepcopy(output_), copy.deepcopy(target_),
                                           self.cfg.beta, thr, self.cfg.metric_type)
            score.append(fscore)
            pre_rec.append([pre, rec])

        score = np.array(score)
        pm = score.argmax()
        best_thr, best_score = thrs[pm], score[pm].item()
        best_pre, best_rec = pre_rec[pm]
        print('thr={} F2={} prec{} rec{}'.format(best_thr, best_score, best_pre, best_rec))
        return best_pre, best_rec, best_score

    def test_step(self):
        """Summary
        Extract the batch of datapoints and return the predicted logits in test step
        Args:
            batch (TYPE): Description
            batch_nb (TYPE): Description

        Returns:
            TYPE: Description
        """
        torch.set_grad_enabled(False)
        self.model.eval()

        output_ = np.array([])
        target_ = np.array([])

        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                if self.hparams.infer == 'valid':
                    # Evaluate
                    inputs, target, names = batch
                    target = target.to(self.device)
                else:
                    # Test
                    inputs, names = batch

                if isinstance(inputs, tuple):
                    inputs = tuple(
                        [e.to(self.device) if type(e) == torch.Tensor else e for e in inputs])
                else:
                    inputs = inputs.to(self.device)

                self.names.extend(names)

                if self.cfg.n_crops:
                    bs, n_crops, c, h, w = inputs.size()
                    inputs = inputs.view(-1, c, h, w)

                logits = self.forward(inputs)

                if self.cfg.n_crops:
                    logits = logits.view(bs, n_crops, -1).mean(1)

                if self.hparams.multi_cls:
                    output = F.softmax(logits)
                    output = output[:, 1]
                else:
                    output = torch.sigmoid(logits)

                if self.hparams.infer == 'valid':
                    target = target.detach().to('cpu').numpy()
                    target_ = np.concatenate((target_, target), axis=0) if len(target_) > 0 else \
                        target
                y_pred = output.detach().to('cpu').numpy()
                output_ = np.concatenate((output_, y_pred), axis=0) if len(output_) > 0 else y_pred

        if self.hparams.infer == 'valid':
            return output_, target_
        else:
            return output_

    def test(self):
        """Summary
        After the test end, calculate the metrics
        Args:

        Returns:
            TYPE: Description
        """
        # inference dataset
        if self.hparams.infer == 'valid':
            output_, target_ = self.test_step()
        else:
            output_ = self.test_step()

        resp = dict()
        to_csv = {'Images': self.names}

        for t in range(len(self.num_tasks)):
            if self.hparams.multi_cls:
                y_pred = np.reshape(output_, output_.shape[0])
            else:
                y_pred = np.transpose(output_)[t]
            to_csv[self.labels[self.num_tasks[t]]] = y_pred

        # Only save scores to json file when in valid mode
        if self.hparams.infer == 'valid':
            overall_pre, overall_rec, overall_fscore = get_metrics(copy.deepcopy(output_),
                                                                   copy.deepcopy(target_),
                                                                   self.cfg.beta,
                                                                   self.cfg.threshold,
                                                                   self.cfg.metric_type)

            resp['overall_pre'] = overall_pre
            resp['overall_rec'] = overall_rec
            resp['overall_f_score'] = overall_fscore

            with open(os.path.join(os.path.dirname(self.hparams.load),
                                   'scores_{}.csv'.format(uuid.uuid4())), 'w') as f:
                json.dump(resp, f)

        # Save predictions to csv file for computing metrics in off-line mode
        path_df = DataFrame(to_csv, columns=to_csv.keys())
        path_df.to_csv(os.path.join(os.path.dirname(self.hparams.load),
                                    'predictions_{}.csv'.format(uuid.uuid4())), index=False)
        return resp

    def configure_optimizers(self):
        """Summary
        Must be implemented
        Returns:
            TYPE: Description
        """
        optimizer = create_optimizer(self.cfg, self.model.parameters())

        if self.cfg.lr_scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg.step_size,
                                                  gamma=self.cfg.lr_factor)
        elif self.cfg.lr_scheduler == 'cosin':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
        elif self.cfg.lr_scheduler == 'cosin_epoch':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.tmax,
                                                             eta_min=self.cfg.eta_min)
        elif self.cfg.lr_scheduler == 'onecycle':
            max_lr = [g["lr"] for g in optimizer.param_groups]
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr,
                                                      epochs=self.hparams.epochs,
                                                      steps_per_epoch=len(self.train_dataloader()))
            scheduler = {"scheduler": scheduler, "interval": "step"}
        else:
            raise ValueError(
                'Does not support {} learning rate scheduler'.format(self.cfg.lr_scheduler))
        return optimizer, scheduler

    def train_dataloader(self):
        """Summary
        Return the train dataset, see dataflow/__init__.py
        Returns:
            TYPE: Description
        """
        ds_train = init_dataset(self.hparams.data_name, cfg=self.cfg,
                                data_path=self.hparams.data_path, mode='train')
        return DataLoader(dataset=ds_train, batch_size=self.cfg.train_batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        """Summary
        Return the val dataset, see dataflow/__init__.py
        Returns:
            TYPE: Description
        """
        ds_val = init_dataset(self.hparams.data_name, cfg=self.cfg,
                              data_path=self.hparams.data_path, mode='valid')
        return DataLoader(dataset=ds_val, batch_size=self.cfg.dev_batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=True)

    def test_dataloader(self):
        """Summary
        Return the test dataset, see dataflow/__init__.py
        Returns:
            TYPE: Description
        """
        ds_test = init_dataset(self.hparams.data_name, cfg=self.cfg,
                               data_path=self.hparams.data_path, mode=self.hparams.infer)
        return DataLoader(dataset=ds_test, batch_size=self.cfg.dev_batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=True)

    def mix_data(self, x, y, device, alpha=1.0):
        """
        Re-constructed input images and labels based on one of two regularization methods such as Mixup and Cutmix.
        :param x: input images
        :param y: labels
        :param device: cpu or gpu device
        :param alpha: parameter for beta distribution
        :return: mixed inputs, pairs of targets, and lambda
        """
        if alpha > 0.:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(device)
        y_a, y_b = y, y[index]

        if self.hparams.mixtype == 'mixup':
            mixed_x = lam * x + (1 - lam) * x[index, :]
        elif self.hparams.mixtype == 'cutmix':
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
            mixed_x = x
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        else:
            raise ValueError('Mixtype {} does not exists'.format(self.hparams.mixtype))

        return mixed_x, y_a, y_b, lam

    def create_scheduler(self, start_epoch):
        """
            Learning rate schedule with respect to epoch
            lr: float, initial learning rate
            lr_factor: float, decreasing factor every epoch_lr
            epoch_now: int, the current epoch
            lr_epochs: list of int, decreasing every epoch in lr_epochs
            return: lr, float, scheduled learning rate.
            """
        count = 0
        for epoch in self.hparams.lr_epochs.split(','):
            if start_epoch >= int(epoch):
                count += 1
                continue

            break

        return self.cfg.lr * np.power(self.cfg.lr_factor, count)

    @staticmethod
    def mixup_criterion(y_a, y_b, lam):
        """
        Re-constructured loss function based on regularization technique
        Args:
            y_a: original labels
            y_b: shuffled labels after random permutation
            lam: generated point in beta distribution
        Returns:
            Combined loss function
        """
        return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    @staticmethod
    def rand_bbox(size, lam):
        """
        Generate random bounding box for specified cutting rate
        Args:
            size: image size including weight and height
            lam: generated point in beta distribution
        Returns:
            Coordinates of top-left and right-bottom vertices of bounding box
        """
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def update_top_k(self, mean, best_dict, metric):
        metric_dev_best = '{}_dev_best'.format(metric)
        metric_top_k = '{}_top_k'.format(metric)
        metric_curr_idx = '{}_curr_idx'.format(metric)

        if metric == 'loss':
            if mean < best_dict[metric_dev_best]:
                best_dict[metric_dev_best] = mean
        else:
            if mean > best_dict[metric_dev_best]:
                best_dict[metric_dev_best] = mean

        if len(best_dict[metric_top_k]) >= self.hparams.save_top_k:
            if metric == 'loss':
                min_idx = best_dict[metric_top_k].index(max(best_dict[metric_top_k]))
            else:
                min_idx = best_dict[metric_top_k].index(min(best_dict[metric_top_k]))
            curr_idx = min_idx
            best_dict[metric_top_k][min_idx] = mean
        else:
            curr_idx = len(best_dict[metric_top_k])
            best_dict[metric_top_k].append(mean)

        best_dict[metric_curr_idx] = curr_idx


def get_args():
    """Summary

    Returns:
        TYPE: Description
    """
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--data_path', metavar='DIR', type=str,
                               default="/vinbrain/data/Vimmec_Data", help='path to dataset')
    parent_parser.add_argument('--json_path', metavar='DIR', type=str, default="cfg.json",
                               help='path to configuration')
    parent_parser.add_argument('--save_path', metavar='DIR', type=str, default="checkpoint",
                               help='path to logging output')
    parent_parser.add_argument('--gpus', type=int, default=0,
                               help='if gpu_id=0, train/test all GPUs, else train/test on specific GPU')
    parent_parser.add_argument('--verbose', action='store_true',
                               help='Detailed info of config file')
    parent_parser.add_argument('--epochs', default=4, type=int, help="Maximum number of epochs")
    parent_parser.add_argument('--seed', type=int, default=0,
                               help='seed for initializing training.')
    parent_parser.add_argument('--num_workers', default=4, type=int,
                               help="Number of workers for each data loader")
    parent_parser.add_argument('--data_name', type=str, default="VinmecDataset", choices=(
        'ChexpertDataset', 'ChexpertDataFlow', 'VinmecDataset', 'VinmecDataFlow', 'LungDataset'),
                               help='dataset name')
    parent_parser.add_argument('--save_top_k', default=5, type=int,
                               help="Save top_k best checkpoints with specific metric")
    parent_parser.add_argument('--cross', type=str, default="",
                               help='Load pre-trained model from other configuration')
    parent_parser.add_argument('--multi_cls', action='store_true',
                               help='Train model for multi-classes or multi-label')
    parent_parser.add_argument('--mixtype', type=str, default='',
                               help='Use mix-up training technique')
    parent_parser.add_argument('--alpha', default=0.2, type=float,
                               help="interpolation strength (uniform=1., ERM=0.)")
    parent_parser.add_argument('--auto_threshold', action='store_true',
                               help='Use early stopping for customized metrics')
    parent_parser.add_argument('--freeze', action='store_true',
                               help='Freeze some layers')
    parent_parser.add_argument('--log-every', default=20, type=int,
                               help='show log console frequently')
    parent_parser.add_argument('--test-every', default=200, type=int,
                               help='validate model frequently')
    parent_parser.add_argument('--metric', default='score', type=str,
                               help='Metric for optimizing model',
                               choices=['score', 'loss'])
    parent_parser.add_argument('--lr-epochs', default='2,4', type=str,
                        help='reduce learning rate by frequency')

    # Inference purpose
    parent_parser.add_argument('--load', type=str, default='',
                               help='load model')
    parent_parser.add_argument('--infer', type=str, default='valid',
                               help='run predict or test by offline')

    return parent_parser.parse_args()


def main(hparams):
    """Summary

    Args:
        hparams (TYPE): Description

    Raises:
        ValueError: Description
    """
    if hparams.seed > 0:
        random.seed(hparams.seed)
        np.random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(hparams.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = MLCModel(hparams)

    if len(hparams.load) > 0 and hparams.infer in ['valid', 'test']:
        # Evaluate or test
        model.test()
    else:
        # Train
        model.train()


if __name__ == '__main__':
    main(get_args())
