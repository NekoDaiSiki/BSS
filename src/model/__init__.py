import importlib

__all__ = ['create_model']


def create_model(opt):
    models_opt = opt.models
    _models = [
        importlib.import_module(f'src.model.{model_file}')
        for model_file in models_opt.model_files
    ]

    main_model_opt = models_opt.main_model
    # sub_models_opt = models_opt.sub_models
    model_cls = None
    for main_model in _models:
        model_cls = getattr(main_model, main_model_opt.type, None)
        if model_cls is not None:
            break

    if model_cls is None:
        raise ValueError(f'Model {main_model_opt.type} not found!')

    return model_cls(opt)