import json
import math
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Union, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from openslide import OpenSlide
from skimage import morphology
from tqdm import tqdm


class DatasetGenerator:
    """Generates dataset consisting of tiles from each slide where the tiles \
    are sampled such that they contain tissue"""
    def __init__(
            self,
            data_dir: Union[str, Path],
            downsampling_tile_width=1024,
            downsampling_tile_height=1024,
            tile_width=1024,
            tile_height=1024,
            fg_thresh=.5,
            use_downsampled_slide=False,
            segmentation_method='otsu'
    ):
        self._data_dir = Path(data_dir)
        self._tile_width = tile_width
        self._tile_height = tile_height
        self._downsampling_tile_width = downsampling_tile_width
        self._downsampling_tile_height = downsampling_tile_height
        self._fg_thresh = fg_thresh
        self._use_downsampled_slide = use_downsampled_slide
        self._segmentation_method = segmentation_method

    @staticmethod
    def _get_downsample_factor_for_slide(slide: OpenSlide, reduction=2e3):
        slide_width, slide_height = slide.dimensions

        downsample_factor = \
            int(min(slide_height / reduction, slide_width / reduction))

        # make power of 2 (necessary due to tile-based downsampling)
        downsample_factor = \
            pow(2, math.floor(math.log(downsample_factor) / math.log(2)))
        return downsample_factor

    def downsample_slide(
            self,
            slide: OpenSlide,
            remove_empty_rows_and_columns=False,
            initial_dim_reduction=2e3,
            pruned_dim_reduction=1e3,
            normalize_background=False
    ) -> np.ndarray:
        """Downsamples slide prior to segmenting in order to fit it in
        memory"""
        slide_width, slide_height = slide.dimensions

        downsample_factor = self._get_downsample_factor_for_slide(
            slide=slide, reduction=initial_dim_reduction)
        tile_resized_width = \
            self._downsampling_tile_width // downsample_factor
        tile_resized_height = \
            self._downsampling_tile_height // downsample_factor

        resized_width = \
            int(int(slide_width / downsample_factor) /
                tile_resized_width) \
            * tile_resized_width
        resized_height = \
            int(int(slide_height / downsample_factor) /
                tile_resized_height) \
            * tile_resized_height

        resized_image = np.zeros((resized_height, resized_width, 3),
                                 dtype='uint8')

        tile_coords = get_tiles_for_slide(
                slide=slide,
                tile_width=self._downsampling_tile_width,
                tile_height=self._downsampling_tile_height
            )
        for x, y in tile_coords:
            region = slide.read_region(
                (x, y), level=0,
                size=(self._downsampling_tile_width,
                      self._downsampling_tile_height))
            resized_region = region.resize((tile_resized_width,
                                            tile_resized_height),
                                           Image.BILINEAR)

            resized_region = np.array(resized_region)
            resized_region = resized_region[:, :, :-1]

            resized_x, resized_y = x // downsample_factor, \
                y // downsample_factor
            resized_image[
                resized_y:resized_y + tile_resized_height,
                resized_x:resized_x + tile_resized_width
            ] = resized_region

        if remove_empty_rows_and_columns:
            mask = self._segment_slide(slide=resized_image)
            if normalize_background:
                resized_image = replace_background_with_constant(
                    img=resized_image, mask=mask)
            pruned = _prune_image_rows_cols(im=resized_image, mask=mask)
            if (np.array(pruned.shape) == 0).any():
                raise NoTissueDetectedError('Bad image. Detected no tissue')
            pruned = Image.fromarray(pruned)
            scale = min(pruned.height / pruned_dim_reduction,
                        pruned.width / pruned_dim_reduction)
            pruned = pruned.resize(
                (int(pruned.width / scale), int(pruned.height / scale)),
                Image.LANCZOS)
            resized_image = np.array(pruned)
        return resized_image

    def _segment_slide(
            self,
            slide: np.ndarray
    ) -> np.ndarray:
        """Segment slide"""
        img_hsv = cv2.cvtColor(slide, cv2.COLOR_BGR2HSV)
        saturation = img_hsv[:, :, 1]

        if self._segmentation_method == 'otsu':
            img_blur = cv2.GaussianBlur(saturation, (5, 5), 0)
            mask = \
                cv2.threshold(img_blur, 0, 1,
                              cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
            mask = mask.astype('bool')
            mask = morphology.remove_small_holes(mask, area_threshold=5000)

        else:
            vals, counts = np.unique(saturation, return_counts=True)
            mode = vals[np.argmax(counts)]
            mask = saturation != mode

        mask = morphology.remove_small_objects(mask, min_size=5000)
        mask = mask.astype('uint8')
        return mask

    def _get_tissue_tiles_from_fullsize_slide(
            self,
            slide: OpenSlide,
            downsampled_slide: np.ndarray,
            mask: np.ndarray,
            tile_coords: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Identifies tiles which contain > fg_thresh foreground pixels
        from the original full size slide"""
        res = []
        downsample_factor = self._get_downsample_factor_for_slide(slide=slide)

        for (x, y) in tile_coords:
            resized_x, resized_y = x // downsample_factor, \
                                   y // downsample_factor

            tile_resized_width = self._tile_width // downsample_factor
            tile_resized_height = self._tile_height // downsample_factor

            is_tissue_tile = self._is_tissue_tile(
                slide=downsampled_slide,
                upper_left_coord=(resized_x, resized_y),
                mask=mask,
                tile_width=tile_resized_width,
                tile_height=tile_resized_height
            )
            if is_tissue_tile:
                res.append((x, y))
        return res

    def _get_tissue_tiles_from_downsampled_slide(
            self,
            slide: np.ndarray,
            mask: np.ndarray,
            tile_coords: List[Tuple[int, int]]):
        res = []
        for (x, y) in tile_coords:
            is_tissue_tile = self._is_tissue_tile(
                slide=slide,
                upper_left_coord=(x, y),
                mask=mask,
                tile_height=self._tile_height,
                tile_width=self._tile_width
            )
            if is_tissue_tile:
                res.append((x, y))
        return res

    def _is_tissue_tile(
            self,
            slide: np.ndarray,
            upper_left_coord: Tuple[int, int],
            mask: np.ndarray,
            tile_width: int,
            tile_height: int
    ):
        x, y = upper_left_coord
        tile_mask = mask[y:y + tile_height, x:x + tile_width]
        is_tissue = (tile_mask != 0)
        foreground_frac = is_tissue.mean()
        if foreground_frac > self._fg_thresh:
            tile = slide[y:y + tile_height, x:x + tile_width]

            tile_gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)

            # avoids noise from being considered foreground pixels
            if np.std(tile_gray) > 5:
                return True
            else:
                return False
        return False

    def _get_tiles_for_slide(
            self,
            image_id: str
    ):
        with OpenSlide(self._data_dir / f'{image_id}.tif') as slide:
            downsampled_slide = self.downsample_slide(
                slide=slide
            )
            mask = self._segment_slide(slide=downsampled_slide)
            tile_coords = get_tiles_for_slide(
                slide=downsampled_slide if self._use_downsampled_slide else slide,
                tile_width=self._tile_width,
                tile_height=self._tile_height
            )

            if self._use_downsampled_slide:
                tissue_tiles = self._get_tissue_tiles_from_downsampled_slide(
                    slide=downsampled_slide,
                    mask=mask,
                    tile_coords=tile_coords
                )
            else:
                tissue_tiles = self._get_tissue_tiles_from_fullsize_slide(
                    slide=slide,
                    downsampled_slide=downsampled_slide,
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


def _prune_image_rows_cols(im, mask, thr=0.01) -> np.ndarray:
    """removes empty rows and columns
    @param im: image
    @param mask: tissue mask
    @param thr: emptiness threshold
    @return: image with empty rows and columns removed

    """
    tissue_rows = mask.mean(axis=1) > thr
    tissue_cols = mask.mean(axis=0) > thr
    im = im[tissue_rows]
    im = im[:, tissue_cols]
    return im


def get_tiles_for_slide(slide: Union[OpenSlide, np.ndarray],
                        tile_width: int, tile_height: int):
    """Gets set of tile upper-left coords for `slide`"""
    slide_width, slide_height = \
        slide.dimensions if isinstance(slide, OpenSlide) else \
        (slide.shape[1], slide.shape[0])

    tile_coords = []
    for y in range(0, slide_height - tile_height + 1,
                   tile_height):
        for x in range(0, slide_width - tile_width + 1,
                       tile_width):
            tile_coords.append((x, y))
    return tile_coords


def replace_background_with_constant(img: np.ndarray, mask: np.ndarray):
    """Replaces background with white"""
    white_bg = np.ones_like(img) * 255
    white_bg[mask == 1] = img[mask == 1]
    return white_bg


class NoTissueDetectedError(RuntimeError):
    pass
