import json
from collections import namedtuple
from pathlib import Path
from typing import Union, List, Optional

import numpy as np
from openslide import OpenSlide
import torch
from torch.utils import data
from torchvision.transforms import transforms
from imgaug import augmenters as iaa

Slide = namedtuple('Slide', ['path', 'target'])
Tile = namedtuple('Tile', ['image_id', 'coords', 'tile_dims', 'slide_idx'])


class MILdataset(data.Dataset):
    def __init__(self, dataset_path: Union[str, Path], mode: str,
                 transform=None):
        valid_modes = ('train', 'inference')
        if mode not in valid_modes:
            raise ValueError(f'Mode must be one of {valid_modes}. Got {mode}')

        self._dataset_path = dataset_path
        self.slides = None
        self.tiles = None
        self.slide_idxs = None
        self.tile_dims = None
        self.construct_dataset()
        self.transform = transform
        self.mode = mode
        self.topk_k = None

    def set_top_k_tiles(
            self,
            tiles: np.ndarray
    ):
        self.topk_k = tiles

    def __getitem__(self, index):
        if self.mode == 'train' and self.topk_k is not None:
            tiles = self.topk_k
        else:
            tiles = self.tiles

        image_id, coords, tile_tims, slide_idx = tiles[index]
        slide_path, target = self.slides[slide_idx]

        with OpenSlide(slide_path) as slide:
            img = slide.read_region(
                location=coords,
                level=0,
                size=tile_tims)\
                .convert('RGB')

            img = np.array(img)

            if self.transform is not None:
                img = self.transform(img)

            return img, target

    def __len__(self):
        if self.mode == 'train' and self.topk_k is not None:
            return len(self.topk_k)
        else:
            return len(self.tiles)

    def construct_dataset(self, downsample: Optional[float] = None):
        with open(self._dataset_path) as f:
            dataset = json.load(f)
        slides = []
        tiles = []
        slide_idx = 0
        for slide in dataset:
            image_id = Path(slide['slide_path']).stem

            tile_coords = slide['tile_coords']
            if downsample is not None:
                tile_coords = np.array(tile_coords)
                idxs = np.arange(len(tile_coords))
                np.random.shuffle(idxs)
                tile_coords = \
                    tile_coords[idxs[:int(downsample * len(tile_coords))]]

            for coord in tile_coords:
                tiles.append(
                    Tile(image_id=image_id,
                         coords=tuple(coord),
                         tile_dims=tuple(slide['tile_dims']),
                         slide_idx=slide_idx))
            if len(tile_coords) > 0:
                slides.append(
                    Slide(path=slide['slide_path'],
                          target=int(slide['target'] == 'LAA')))
                slide_idx += 1

        self.slides = np.array(slides, dtype='object')
        self.tiles = np.array(tiles, dtype='object')
        self.slide_idxs = np.array(
            [slide_idx for _, _, _, slide_idx in tiles])
        self.tile_dims = tiles[0].tile_dims


def get_dataloader(dataset_path: Union[str, Path],
                   mode,
                   batch_size=512,
                   n_workers=4):
    train_agumentations = [
        iaa.Sequential([
            iaa.Rotate(
                rotate=[0, -90, -180, -270, 270, 180, 90]
            ),
            iaa.Fliplr(p=0.5),
            iaa.Flipud(p=0.5),
        ]).augment_image,
        np.copy
    ]

    all_augmentations = [
        transforms.ToTensor(),
        # # TODO use imagenet stats?
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
    ]

    if mode == 'train':
        augmentations = train_agumentations + all_augmentations
    else:
        augmentations = all_augmentations
    trans = transforms.Compose(augmentations)

    # load data
    dset = MILdataset(dataset_path=dataset_path, transform=trans, mode=mode)

    data_loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size, shuffle=mode == 'train',
        num_workers=n_workers, pin_memory=False)
    return data_loader
