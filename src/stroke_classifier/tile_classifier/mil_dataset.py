import json
from collections import namedtuple
from pathlib import Path
from typing import Union, List

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

        with open(dataset_path) as f:
            dataset = json.load(f)
        slides = []
        tiles = []
        for i, slide in enumerate(dataset):
            image_id = Path(slide['slide_path']).stem
            slides.append(
                Slide(path=slide['slide_path'],
                      target=int(slide['target'] == 'LAA')))

            for coord in slide['tile_coords']:
                tiles.append(
                    Tile(image_id=image_id,
                         coords=tuple(coord),
                         tile_dims=tuple(slide['tile_dims']),
                         slide_idx=i))

        self.slides = np.array(slides, dtype='object')
        self.tiles = np.array(tiles, dtype='object')
        self.transform = transform
        self.mode = mode
        self.topk_k = None

    def set_top_k_tiles(
            self,
            top_k_indices: List[int]
    ):
        self.topk_k = [self.tiles[idx] for idx in top_k_indices]

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
