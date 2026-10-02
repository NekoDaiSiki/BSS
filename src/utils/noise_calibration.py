"""Fit a shared affine variance shape from noisy-only local statistics.

The profiled objective follows Main-R1's noisy-only calibration: each group
has its own nonnegative scale; groups have equal weight after normalizing
their weighted cell MSE by squared mean observed variance. This module uses
only the Python standard library so fitting can run separately from inference.
"""

import math
from collections import defaultdict


def fit_affine_shape(rows, min_pixels=32, grid_size=1025):
    """Fit variance = c_group * (a * mean + b), a+b=1, a,b>=0.

    Rows contain group, mean, variance, count. Variance must be a centered
    *sample* variance of the selected residuals; its weight is count - 1.
    The returned epsilon is b/a, or None at the constant-variance endpoint.
    """
    if min_pixels < 2 or grid_size < 3:
        raise ValueError('Require min_pixels >= 2 and grid_size >= 3')
    groups = defaultdict(list)
    rejected = 0
    for row in rows:
        x, y, count = (float(row[key]) for key in ('mean', 'variance', 'count'))
        if (not all(math.isfinite(v) for v in (x, y, count))
                or not 0 <= x <= 1 or y < 0 or count < min_pixels
                or not count.is_integer()):
            rejected += 1
            continue
        groups[str(row['group'])].append((x, y, count - 1))

    moments = {}
    skipped_groups = []
    for group, cells in groups.items():
        w = math.fsum(z for _, _, z in cells)
        sx = math.fsum(z * x for x, _, z in cells)
        sxx = math.fsum(z * x * x for x, _, z in cells)
        sy = math.fsum(z * y for _, y, z in cells)
        sxy = math.fsum(z * x * y for x, y, z in cells)
        syy = math.fsum(z * y * y for _, y, z in cells)
        if sy <= 0 or len(cells) < 2:
            skipped_groups.append(group)
            continue
        moments[group] = (w, sx, sxx, sy, sxy, syy)
    if not moments:
        raise ValueError('No groups have enough valid cells with positive variance')
    if max(sxx / w - (sx / w) ** 2 for w, sx, sxx, _, _, _ in moments.values()) < 1e-12:
        raise ValueError('Insufficient within-group intensity variation to identify epsilon')

    def evaluate(a, details=False):
        b = 1.0 - a
        errors, scales = [], {}
        for group, (w, sx, sxx, sy, sxy, syy) in moments.items():
            numerator = a * sxy + b * sy
            denominator = a * a * sxx + 2 * a * b * sx + b * b * w
            scale = max(0.0, numerator / denominator) if denominator > 0 else 0.0
            # Clamp tiny negative SSE caused by cancellation at an exact fit.
            sse = max(0.0, syy - 2 * scale * numerator + scale * scale * denominator)
            error = (sse / w) / (sy / w) ** 2
            errors.append(error)
            scales[group] = {'scale': scale, 'relative_rmse': math.sqrt(error),
                             'cells': len(groups[group]), 'weight': w}
        loss = math.fsum(errors) / len(errors)
        return (loss, scales) if details else loss

    # Scan the whole interval, then refine every bracketed local minimum.
    # A single bounded search need not find the best minimum of a mixture of
    # group-level profiled errors. Endpoints remain explicit candidates.
    grid = [i / (grid_size - 1) for i in range(grid_size)]
    values = [evaluate(a) for a in grid]
    candidates = [(values[i], grid[i]) for i in (0, grid_size - 1)]
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    for i in range(1, grid_size - 1):
        if values[i] <= values[i - 1] and values[i] <= values[i + 1]:
            lo, hi = grid[i - 1], grid[i + 1]
            left, right = hi - golden * (hi - lo), lo + golden * (hi - lo)
            fl, fr = evaluate(left), evaluate(right)
            for _ in range(64):
                if hi - lo < 1e-12:
                    break
                if fl <= fr:
                    hi, right, fr = right, left, fl
                    left = hi - golden * (hi - lo)
                    fl = evaluate(left)
                else:
                    lo, left, fl = left, right, fr
                    right = lo + golden * (hi - lo)
                    fr = evaluate(right)
            a = (lo + hi) / 2
            candidates.append((evaluate(a), a))
            candidates.append((values[i], grid[i]))
    loss, a = min(candidates)
    # Prefer an exact endpoint only when it is numerically tied with the fit.
    for endpoint in (0.0, 1.0):
        endpoint_loss = evaluate(endpoint)
        if endpoint_loss <= loss + 1e-14:
            loss, a = endpoint_loss, endpoint
            break
    loss, scales = evaluate(a, details=True)
    return {
        'a': a, 'b': 1.0 - a,
        'epsilon': (1.0 - a) / a if a > 0 else None,
        'objective': loss,
        'groups': scales,
        'accepted_cells': sum(len(groups[group]) for group in moments),
        'rejected_cells': rejected,
        'skipped_groups': skipped_groups,
        'variance_estimator': 'centered sample variance (N-1)',
        'weighting': 'count-1 within groups; normalized MSE equally across groups',
    }


def calibrated_training_config(config, fit):
    """Return a separate stdnorm training config using the fitted shape."""
    import copy

    result = copy.deepcopy(config)
    params = result['train']['loss_params']
    a, b = fit['a'], fit['b']
    if a == 0:
        result['train']['mode'] = 'v4_stdmask_stdnorm'
        params.pop('norm_bias', None)
        params.pop('norm_weight', None)
    elif b == 0:
        result['train']['mode'] = 'v4_stdmask_poinorm_stdnorm'
        params.pop('norm_bias', None)
        params['norm_weight'] = 1
    else:
        result['train']['mode'] = 'v4_stdmask_poinorm_stdnorm_v2'
        params['norm_bias'] = fit['epsilon']
        params.pop('norm_weight', None)
    params.pop('norm_type', None)
    result['task_name'] += '_estimated'
    result['description'] = 'BSS stdnorm with an affine variance shape fitted from noisy training residuals'
    return result
