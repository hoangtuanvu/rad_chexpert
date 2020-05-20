"""Summary
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .dataset import XrayDataset

__factory = {
    'XrayDataset': XrayDataset,
}


def init_labels(name='XrayDataset', **kwargs):
    """Summary
    
    Args:
        name (str, optional): Chexpert or Vinmec, default is Chexpert
        **kwargs: Description
    
    Returns:
        TYPE: Description
    
    Raises:
        KeyError: Description
    """
    all_datasets = list(__factory.keys())
    if name.startswith("XrayDataset"):
        __labels = ['No_Finding',
                    'Enlarged_Cardiomediastinum',
                    'Cardiomegaly',
                    'Lung_Opacity',
                    'Lung_Lesion',
                    'Edema',
                    'Consolidation',
                    'Pneumonia',
                    'Atelectasis',
                    'Pneumothorax',
                    'Pleural_Effusion',
                    'Pleural_Other',
                    'Fracture',
                    'Support_Devices']
    else:
        raise KeyError('Invalid dataset name. Received "{}", but expected to be one of {}'.format(name, all_datasets))
    return __labels


def init_dataset(name, **kwargs):
    """Summary
    
    Args:
        name (TYPE): Description
        **kwargs: Description
    
    Returns:
        TYPE: Description
    
    Raises:
        KeyError: Description
    """
    all_datasets = list(__factory.keys())
    if name not in all_datasets:
        raise KeyError('Invalid dataset name. Received "{}", but expected to be one of {}'.format(name, all_datasets))
    return __factory[name](**kwargs)
