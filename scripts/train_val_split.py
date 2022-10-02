import argparse
import json

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--train_tiles_path', required=True)
parser.add_argument('--out_path', required=True)

args = parser.parse_args()


def main():
    with open(args.train_tiles_path) as f:
        train_tiles = json.load(f)

    train_tiles = [x for x in train_tiles if len(x['tile_coords']) > 0]

    rng = np.random.default_rng(1234)
    idxs = np.arange(len(train_tiles))
    rng.shuffle(idxs)
    train_frac = 0.7

    train = [train_tiles[idx] for idx in
             idxs[:int(len(train_tiles) * train_frac)]]
    val = [train_tiles[idx] for idx in
           idxs[int(len(train_tiles) * train_frac):]]

    with open(f'{args.out_path}/train_tiles.json', 'w') as f:
        f.write(json.dumps(train, indent=2))

    with open(f'{args.out_path}/val_tiles.json', 'w') as f:
        f.write(json.dumps(val, indent=2))


if __name__ == '__main__':
    main()
