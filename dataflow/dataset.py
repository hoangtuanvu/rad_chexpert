"""Summary
"""
import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
from .transforms import data_augment, gray_process


class XrayDataset(Dataset):
    """
    Image Pre-Processing for all labeled datasets
    
    Attributes:
        cfg (TYPE): Description
        df (TYPE): Description
        imagePaths (TYPE): Description
        labels (TYPE): Description
        mode (TYPE): Description
    """

    def __init__(self, cfg, data_path=None, mode='train'):
        """Summary
        
        Args:
            cfg (TYPE): Description
            data_path (None, optional): Description
            mode (str, optional): Description
        """
        super(XrayDataset, self).__init__()
        self.cfg = cfg
        self.mode = mode
        self.data_path = data_path

        if mode == 'train':
            self.df = pd.read_csv(os.path.join(data_path, '{}.csv'.format(self.cfg.train_name)))
        elif mode == 'valid':
            self.df = pd.read_csv(os.path.join(data_path, '{}.csv'.format(self.cfg.valid_name)))
        else:
            self.df = pd.read_csv(os.path.join(data_path, '{}.csv'.format(mode)))

        self.df = self.df.fillna(0)

        if self.cfg.agree_rate > 0:
            self.df[list(self.df)[1:]] = 1 * (self.df[list(self.df)[1:]] >= self.cfg.agree_rate)

        self.imagePaths = list()
        self.labels = list()

        zero_cnt = 0
        ones_cnt = 0
        for i, row in self.df.iterrows():
            path = os.path.join(data_path, self.cfg.source_name, row['Images'])

            if os.path.exists(path) and 'lateral' not in path:
                if self.mode == 'train':
                    if len(self.cfg.extract_fields.split(',')) == 1:
                        idx = int(self.cfg.extract_fields) + self.cfg.offset

                        is_added = False
                        if row[idx] == 0:
                            zero_cnt += 1

                            if zero_cnt < self.cfg.zero_limits:
                                is_added = True
                        elif row[idx] == 1:
                            ones_cnt += 1

                            if ones_cnt < self.cfg.ones_limits:
                                is_added = True

                        if is_added:
                            self.labels.append(row[idx])
                            self.imagePaths.append(path)
                    else:
                        idx = list(map(int, self.cfg.extract_fields.split(',')))
                        idx = [index + self.cfg.offset for index in idx]
                        self.labels.append(list(row[idx].values))
                        self.imagePaths.append(path)

                    flg_upsample = False
                    for disease_idx in self.cfg.upsample_index:
                        if isinstance(idx, list):
                            if (disease_idx + self.cfg.offset) in idx:
                                flg_upsample = True
                                break
                        else:
                            if disease_idx + self.cfg.offset == idx:
                                flg_upsample = True
                                break

                    if flg_upsample and self.mode == 'train':
                        for j in range(self.cfg.upsample_times):
                            if len(self.cfg.extract_fields.split(',')) == 1:
                                if is_added:
                                    self.labels.append(row[idx])
                                    self.imagePaths.append(path)
                            else:
                                self.labels.append(list(row[idx].values))
                                self.imagePaths.append(path)
                else:
                    self.imagePaths.append(path)

                    if self.mode == 'valid':
                        if len(self.cfg.extract_fields.split(',')) == 1:
                            idx = int(self.cfg.extract_fields) + self.cfg.offset
                            self.labels.append(row[idx])
                        else:
                            idx = list(map(int, self.cfg.extract_fields.split(',')))
                            idx = [index + self.cfg.offset for index in idx]
                            self.labels.append(list(row[idx].values))

    def __getitem__(self, index):
        """Summary

        Args:
            index (TYPE): Description

        Returns:
            TYPE: Description
        """
        img_path = self.imagePaths[index]
        img_name = img_path[len("{}/{}".format(self.data_path, self.cfg.source_name)):]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if not self.cfg.gray:
            img = data_augment(img=img, cfg=self.cfg, mode=self.mode)
        else:
            img = gray_process(img=img, cfg=self.cfg, mode=self.mode)

        if self.mode == 'train' or self.mode == 'valid':
            if len(self.cfg.extract_fields.split(',')) > 1:
                if self.cfg.criterion == 'class_balance':
                    return img, torch.LongTensor(self.labels[index]), img_name
                else:
                    return img, torch.FloatTensor(self.labels[index]), img_name
            else:
                return img, int(self.labels[index]), img_name
        else:
            return img, img_name

    def __len__(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        return len(self.imagePaths)
