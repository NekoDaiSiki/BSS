"""Epoch-based loss changes in a continuous training run."""


def validate_mean_loss_stages(train_opt):
    stages = train_opt.get('mean_loss_stages')
    if not stages:
        return ()

    expected_start = 0
    total_epoch = train_opt['total_epoch']
    for stage in stages:
        start = stage['start_epoch']
        end = stage['end_epoch']
        weight = stage['weight']
        ws = stage['ws']
        if start != expected_start or end <= start or end > total_epoch:
            raise ValueError('mean_loss_stages must cover consecutive epochs from 0')
        if weight < 0 or not isinstance(ws, int) or ws < 1 or ws % 2 != 1:
            raise ValueError('mean_loss_stages require a nonnegative weight and odd ws')
        if 'weight_end' in stage and stage['weight_end'] < weight:
            raise ValueError('weight_end must be at least the starting weight')
        if 'lower' in stage and stage['lower'] < 0:
            raise ValueError('mean_loss_stages require a nonnegative lower')
        if 'thres' in stage and stage['thres'] < 0:
            raise ValueError('mean_loss_stages require a nonnegative thres')
        expected_start = end

    if expected_start != total_epoch:
        raise ValueError('mean_loss_stages must cover every training epoch')
    return stages


def mean_loss_stage_for_epoch(stages, epoch):
    for stage in stages:
        if stage['start_epoch'] <= epoch < stage['end_epoch']:
            return stage
    raise ValueError('No mean loss stage covers epoch {}'.format(epoch))


def apply_mean_loss_stage(loss_params, stage, epoch):
    mean_loss_params = loss_params['mean_loss_params']
    weight = stage['weight']
    if 'weight_end' in stage:
        progress = (epoch - stage['start_epoch'] + 1) / (
            stage['end_epoch'] - stage['start_epoch']
        )
        weight += (stage['weight_end'] - weight) * progress
    mean_loss_params['weight'] = weight
    mean_loss_params['ws'] = stage['ws']
    if 'thres' in stage:
        mean_loss_params['thres'] = stage['thres']
    if 'lower' in stage:
        loss_params['lower'] = stage['lower']
    return weight
