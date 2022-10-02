import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--train_slides_path', required=True)
parser.add_argument('--train_slides_meta_path', required=True)
parser.add_argument('--out_path', required=True)
parser.add_argument('--train_frac', default=0.7, type=float)

args = parser.parse_args()


def main():
    with open(args.train_slides_meta_path) as f:
        train_slides = json.load(f)

    train_slides = [x for x in train_slides if len(x['tile_coords']) > 0]

    for slide in train_slides:
        slide_path = Path(slide['slide_path'])
        # Update slide path to new path
        slide['slide_path'] = \
            str(Path(f'{args.train_slides_path}') / slide_path.stem)

    rng = np.random.default_rng(1234)
    idxs = np.arange(len(train_slides))
    rng.shuffle(idxs)
    train_frac = args.train_frac

    train = [train_slides[idx] for idx in
             idxs[:int(len(train_slides) * train_frac)]]
    val = [train_slides[idx] for idx in
           idxs[int(len(train_slides) * train_frac):]]

    with open(f'{args.out_path}/train_tiles.json', 'w') as f:
        f.write(json.dumps(train, indent=2))

    with open(f'{args.out_path}/val_tiles.json', 'w') as f:
        f.write(json.dumps(val, indent=2))

    print(f'Train slides: {len(train)}')
    print(f'Val slides: {len(val)}')


if __name__ == '__main__':
    main()
