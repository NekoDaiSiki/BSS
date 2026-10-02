"""Prepare the 512x512 noisy SIDD raw MAT crops used by the raw configurations."""
import argparse
from pathlib import Path
import h5py
import numpy as np
from scipy.io import savemat


def crop_file(path, output):
    with h5py.File(path, 'r') as handle:
        raw = handle['x']
        height, width = raw.shape
        if min(height, width) < 512:
            raise ValueError('Raw image is smaller than 512 pixels: {}'.format(path))
        rows = list(range(0, height - 512 + 1, 256))
        columns = list(range(0, width - 512 + 1, 256))
        if rows[-1] != height - 512:
            rows.append(height - 512)
        if columns[-1] != width - 512:
            columns.append(width - 512)
        index = 0
        for row in rows:
            for column in columns:
                index += 1
                crop = np.ascontiguousarray(raw[row:row + 512, column:column + 512])
                savemat(str(output / '{}_s{:03d}.mat'.format(path.stem, index)), {'x': crop})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('datasets/train/SIDD_Medium_Raw_noisy_sub512'))
    args = parser.parse_args()
    files = sorted(args.input_dir.rglob('*NOISY*.MAT'))
    if not files:
        parser.error('No noisy raw MAT files found')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Output directory must be empty')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        crop_file(path, args.output_dir)
    print('Prepared crops from {} noisy images'.format(len(files)))


if __name__ == '__main__':
    main()
