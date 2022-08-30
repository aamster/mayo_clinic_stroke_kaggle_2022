import random
import sys
from pathlib import Path
from typing import Union, List

import numpy as np
import openslide
import torch
from PIL import Image
from torch.utils import data
from torchvision.transforms import transforms


class MILdataset(data.Dataset):
    def __init__(self, dataset_path: Union[str, Path], mode: str,
                 transform=None):
        valid_modes = ('train', 'inference')
        if mode not in valid_modes:
            raise ValueError(f'Mode must be one of {valid_modes}. Got {mode}')

        dataset = torch.load(dataset_path)
        slides = []
        for i, name in enumerate(dataset['slides']):
            sys.stdout.write(
                f'Opening SVS headers: [{i+1}/{dataset["slides"]}]\r')
            sys.stdout.flush()
            slides.append({
                'index': i,
                'slide': openslide.OpenSlide(name),
                'path': name,
                'target': dataset['targets'][i]
            })
        print('')

        # Flatten grid
        tiles = []
        for i, slide_tiles in enumerate(dataset['grid']):
            tiles += [
                {'slide': i,
                 'coord': tile,
                 'target': (dataset['targets'][i]
                            if 'targets' in dataset else None)
                 } for tile in slide_tiles]

        print('Number of tiles: {}'.format(len(tiles)))

        self.slides = slides
        self.tiles = tiles
        self.transform = transform
        self.mode = mode
        self.topk_k = None
        self.mult = dataset['mult']
        self.size = int(np.round(224 * dataset['mult']))
        self.level = dataset['level']

    def get_top_k_tiles(
            self,
            top_k_indices: List[int]
    ):
        data = []
        for idx in top_k_indices:
            slide_idx, coords, target = self.tiles[idx]
            data.append((slide_idx, coords, target))
        self.topk_k = data

    def __getitem__(self, index):
        dataset = self.topk_k if self.mode == 'train' else self.tiles
        slide_idx, coord, target = dataset[index]

        slide = self.slides[slide_idx]['slide']
        img = slide.read_region(coord, self.level, (self.size, self.size))\
            .convert('RGB')
        if self.mult != 1:
            img = img.resize((self.size, self.size), Image.BILINEAR)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        if self.mode == 'train':
            return len(self.topk_k)
        else:
            return len(self.tiles)


def get_dataloader(dataset_path: Union[str, Path],
                   mode,
                   batch_size=512,
                   n_workers=4):
    # normalization
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    # load data
    dset = MILdataset(dataset_path=dataset_path, transform=trans, mode=mode)

    data_loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size, shuffle=mode == 'train',
        num_workers=n_workers, pin_memory=False)
    return data_loader
