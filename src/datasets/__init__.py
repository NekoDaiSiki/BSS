"""Dataset classes used by the maintained training and test configurations."""

import importlib

DATASET_MODULES = {
    'ImagenetValDataset': 'srgb_datasets',
    'BasicValidateDataset': 'srgb_datasets',
    'DataLoader_SIDD_Medium_Raw': 'raw_datasets',
    'DataLoader_SIDD_Medium_Raw_mod2': 'raw_datasets',
    'Val_SIDD_Medium_Raw': 'raw_datasets',
    'SIDDSrgbTrainDataset_crop': 'SA_sidd_srgb_datasets',
    'SIDDSrgbValidationDataset': 'SA_sidd_srgb_datasets',
}


def create_dataset(dataset_opt):
    name = dataset_opt.type
    if name not in DATASET_MODULES:
        raise ValueError('Unknown dataset: {}'.format(name))
    module = importlib.import_module('src.datasets.' + DATASET_MODULES[name])
    return getattr(module, name)(dataset_opt)
