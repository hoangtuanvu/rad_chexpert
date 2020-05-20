"""Summary
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from dataflow import augmentations
import kornia as K


def augment_and_mix(image, trans, cfg):
    """Perform AugMix augmentations and compute mixture.
      Args:
        image: PIL.Image input image
        preprocess: Preprocessing function which should return a torch tensor.
      Returns:
        mixed: Augmented and mixed image.
      """
    aug_list = augmentations.augmentations
    if cfg.all_ops:
        aug_list = augmentations.augmentations_all

    ws = np.float32(np.random.dirichlet([cfg.aug_prob_coeff] * cfg.mixture_width))
    m = np.float32(np.random.beta(cfg.aug_prob_coeff, cfg.aug_prob_coeff))

    mix = torch.zeros_like(trans(image))
    for i in range(cfg.mixture_width):
        image_aug = image.copy()
        depth = cfg.mixture_depth if cfg.mixture_depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, cfg.aug_severity)

        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * trans(image_aug)

    mixed = (1 - m) * trans(image) + m * mix
    return mixed


def gray_process(img, cfg, mode='train'):
    def _normalize(sample, maxval):
        """Scales images to be roughly [-1024 1024]."""
        sample = (2 * (sample.astype(np.float32) / maxval) - 1.) * 1024
        return sample

    img = _normalize(img, 255)

    # Check that images are 2D arrays
    if len(img.shape) > 2:
        img = img[:, :, 0]
    if len(img.shape) < 2:
        print("error, dimension lower than 2 for image")

    # Add color channel
    img = img[None, :, :]

    if mode == 'train':
        transform = nn.Sequential(augmentations.ToTensor(), augmentations.XRayResizer(cfg.img_size),
                                  K.augmentation.CenterCrop(size=cfg.crop_size),
                                  augmentations.XrayRandomHorizontalFlip(p=0.5),
                                  augmentations.Squeeze(), )
    else:
        transform = nn.Sequential(augmentations.ToTensor(), augmentations.XRayResizer(cfg.img_size),
                                  K.augmentation.CenterCrop(size=cfg.crop_size),
                                  augmentations.Squeeze())

    return transform(img)


def data_augment(img, cfg, mode='train'):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = Image.fromarray(img)

    if cfg.imagenet:
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize(
            [cfg.pixel_mean / 256, cfg.pixel_mean / 256, cfg.pixel_mean / 256],
            [cfg.pixel_std / 256, cfg.pixel_std / 256, cfg.pixel_std / 256])

    if mode == 'train':
        if cfg.n_crops == 10:
            trans = transforms.Compose(
                [transforms.RandomResizedCrop(size=cfg.img_size, scale=(0.8, 1.0)),
                 transforms.RandomRotation(degrees=5), transforms.ColorJitter(),
                 transforms.TenCrop(size=cfg.crop_size), transforms.Lambda(
                    lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                 transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))])
        elif cfg.n_crops == 5:
            trans = transforms.Compose(
                [transforms.RandomResizedCrop(size=cfg.img_size, scale=(0.8, 1.0)),
                 transforms.RandomRotation(degrees=5), transforms.ColorJitter(),
                 transforms.RandomHorizontalFlip(), transforms.FiveCrop(size=cfg.crop_size),
                 transforms.Lambda(
                     lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                 transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))])
        else:
            trans = transforms.Compose(
                [transforms.Resize(size=cfg.img_size), transforms.CenterCrop(size=cfg.crop_size),
                 transforms.RandomRotation(degrees=5), transforms.ColorJitter(),
                 transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    elif mode == 'test' and cfg.n_crops > 0:
        trans = transforms.Compose(
            [transforms.Resize(size=cfg.img_size), transforms.FiveCrop(size=cfg.crop_size),
             transforms.Lambda(
                 lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
             transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))])
    else:
        trans = transforms.Compose(
            [transforms.Resize(size=cfg.img_size), transforms.CenterCrop(size=cfg.crop_size),
             transforms.ToTensor(), normalize])

    if mode == 'train' and cfg.n_crops == 0 and cfg.augmix:
        if cfg.no_jsd:
            return augment_and_mix(img, trans, cfg)
        else:
            return trans(img), augment_and_mix(img, trans, cfg), augment_and_mix(img, trans, cfg)
    else:
        return trans(img)
