import json
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Union, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from openslide import OpenSlide
from tqdm import tqdm


class DatasetGenerator:
    """Generates dataset consisting of tiles from each slide where the tiles \
    are sampled such that they contain tissue"""
    def __init__(
            self,
            data_dir: Union[str, Path],
            tile_width=1024,
            tile_height=1024,
            fg_thresh=.5,
            downsample_factor=64
    ):
        self._data_dir = Path(data_dir)
        self._tile_width = tile_width
        self._tile_height = tile_height
        self._fg_thresh = fg_thresh
        self._downsample_factor = downsample_factor

    def downsample_slide(
            self,
            slide: OpenSlide,
            tile_coords: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Downsamples slide prior to segmenting in order to fit it in
        memory"""
        slide_width, slide_height = slide.dimensions

        tile_resized_width = self._tile_width // self._downsample_factor
        tile_resized_height = self._tile_height // self._downsample_factor

        resized_width = \
            int(int(slide_width / self._downsample_factor) /
                tile_resized_width) \
            * tile_resized_width
        resized_height = \
            int(int(slide_height / self._downsample_factor) /
                tile_resized_height) \
            * tile_resized_height

        resized_image = np.zeros((resized_height, resized_width, 3),
                                 dtype='uint8')

        for x, y in tile_coords:
            region = slide.read_region(
                (x, y), level=0,
                size=(self._tile_width, self._tile_height))
            resized_region = region.resize((tile_resized_width,
                                            tile_resized_height),
                                           Image.BILINEAR)

            resized_region = np.array(resized_region)
            resized_region = resized_region[:, :, :-1]

            resized_x, resized_y = x // self._downsample_factor, \
                y // self._downsample_factor
            resized_image[
                resized_y:resized_y + tile_resized_height,
                resized_x:resized_x + tile_resized_width
            ] = resized_region
        return resized_image

    def _segment_slide(
            self,
            slide: np.ndarray
    ) -> np.ndarray:
        """Segment slide by applying otsu"""
        img_gray = cv2.cvtColor(slide, cv2.COLOR_RGB2GRAY)

        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        mask = \
            cv2.threshold(img_blur, 0, 1,
                          cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]
        return mask

    def _get_tissue_tiles(
            self,
            slide: OpenSlide,
            mask: np.ndarray,
            tile_coords: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Identifies tiles which contain > fg_thresh foreground pixels"""
        res = []
        for (x, y) in tile_coords:
            resized_x, resized_y = x // self._downsample_factor, \
                                   y // self._downsample_factor

            tile_resized_width = self._tile_width // self._downsample_factor
            tile_resized_height = self._tile_height // self._downsample_factor

            tile_mask = mask[
                resized_y:resized_y + tile_resized_height,
                resized_x:resized_x + tile_resized_width]

            is_tissue = (tile_mask != 0)
            foreground_frac = is_tissue.mean()
            if foreground_frac > self._fg_thresh:
                tile = slide.read_region(
                    location=(x, y),
                    level=0,
                    size=(self._tile_width, self._tile_height)
                )
                tile = np.array(tile)
                tile_gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)

                # avoids noise from being considered foreground pixels
                if np.std(tile_gray) > 5:
                    res.append((x, y))
        return res

    def _get_tiles_for_slide(self, image_id: str):
        with OpenSlide(self._data_dir / f'{image_id}.tif') as slide:
            slide_width, slide_height = slide.dimensions

            tile_coords = []
            for y in range(0, slide_height - self._tile_height,
                           self._tile_height):
                for x in range(0, slide_width - self._tile_width,
                               self._tile_width):
                    tile_coords.append((x, y))
            downsampled_slide = self.downsample_slide(
                slide=slide,
                tile_coords=tile_coords
            )
            mask = self._segment_slide(slide=downsampled_slide)
            tissue_tiles = self._get_tissue_tiles(
                slide=slide,
                mask=mask,
                tile_coords=tile_coords
            )
        return tissue_tiles

    def get_tiles(
            self,
            meta_path: Union[str, Path],
            out_path: Union[str, Path]
    ):
        meta_path = Path(meta_path)

        meta = pd.read_csv(meta_path, dtype={'center_id': str})
        image_ids = meta['image_id'].tolist()
        targets = meta['label'].tolist()
        with Pool(processes=os.cpu_count()) as p:
            res = list(tqdm(
                p.imap(self._get_tiles_for_slide, image_ids),
                total=len(image_ids),
                desc='Getting tiles'
            ))

        tiles = [{
            'slide_path': str(self._data_dir / f'{image_ids[i]}.tif'),
            'tile_coords': res[i],
            'tile_dims': (self._tile_width, self._tile_height),
            'target': targets[i]
        } for i in range(len(image_ids))]

        with open(out_path, 'w') as f:
            json.dump(tiles, f, indent=2)
