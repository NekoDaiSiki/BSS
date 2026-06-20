import os
import glob
import importlib

__all__ = ['create_dataset']


def create_dataset(dataset_opt):
    # fix the datasets path
    dataset_files = os.listdir('./src/datasets/')
    _datasets = [
        importlib.import_module(f'src.datasets.{os.path.splitext(dataset_file)[0]}')
        for dataset_file in dataset_files]

    dataset_cls = None
    for dataset in _datasets:
        dataset_cls = getattr(dataset, dataset_opt.type, None)
        if dataset_cls is not None:
            break

    if dataset_cls is None:
        raise ValueError(f'Model {dataset_opt.type} not found!')

    return dataset_cls(dataset_opt)