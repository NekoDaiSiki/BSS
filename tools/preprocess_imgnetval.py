"""Prepare RGB ImageNet images with the original 256-512 pixel size filter."""
import argparse
from pathlib import Path
from PIL import Image


def prepare_image(path, output):
    with Image.open(path) as image:
        if not (256 <= image.width <= 512 and 256 <= image.height <= 512):
            return 0
        image.convert('RGB').save(output / path.name, quality=100, subsampling=0)
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('datasets/train/Imagenet_val'))
    args = parser.parse_args()
    files = sorted(path for path in args.input_dir.rglob('*') if path.suffix.lower() in {'.jpeg', '.jpg', '.png'})
    if not files:
        parser.error('No input images found')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Output directory must be empty')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    count = sum(prepare_image(path, args.output_dir) for path in files)
    print('Prepared {} RGB images'.format(count))


if __name__ == '__main__':
    main()
