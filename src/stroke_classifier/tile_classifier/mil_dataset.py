import json
from pathlib import Path
from typing import Union, List

from openslide import OpenSlide
import torch
from torch.utils import data
from torchvision.transforms import transforms
from imgaug import augmenters as iaa


class MILdataset(data.Dataset):
    def __init__(self, dataset_path: Union[str, Path], mode: str,
                 transform=None):
        valid_modes = ('train', 'inference')
        if mode not in valid_modes:
            raise ValueError(f'Mode must be one of {valid_modes}. Got {mode}')

        with open(dataset_path) as f:
            dataset = json.load(f)
        slides = {}
        tiles = []
        for i, slide in enumerate(dataset):
            image_id = Path(slide['slide_path']).stem
            slides[image_id] = {
                'path': slide['slide_path'],
                'target': int(slide['target'] == 'LAA'),
                'index': i
            }
            for coord in slide['tile_coords']:
                tiles.append({
                    'image_id': image_id,
                    'coords': coord,
                    'dims': slide['tile_dims']
                })

        self.slides = slides
        self.tiles = tiles
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

        tile = tiles[index]

        with OpenSlide(self.slides[tile['image_id']]['path']) as slide:
            img = slide.read_region(
                location=tile['coords'],
                level=0,
                size=tile['dims'])\
                .convert('RGB')

            if self.transform is not None:
                img = self.transform(img)

            return img, self.slides[tile['image_id']]['target']

    def __len__(self):
        if self.mode == 'train' and self.topk_k is not None:
            return len(self.topk_k)
        else:
            return len(self.tiles)


def get_dataloader(dataset_path: Union[str, Path],
                   mode,
                   batch_size=512,
                   n_workers=4):
    trans = transforms.Compose([
        iaa.Rotate(
            rotate=[0, -90, -180, -270, 270, 180, 90]
        ),
        iaa.Fliplr(p=0.5),
        iaa.Flipud(p=0.5),
        transforms.ToTensor(),
        # TODO use imagenet stats?
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
    ])

    # load data
    dset = MILdataset(dataset_path=dataset_path, transform=trans, mode=mode)

    data_loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size, shuffle=mode == 'train',
        num_workers=n_workers, pin_memory=torch.cuda.is_available())
    return data_loader
