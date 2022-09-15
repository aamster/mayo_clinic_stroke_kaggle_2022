from pathlib import Path

import cv2
import numpy as np
from openslide import OpenSlide

from stroke_classifier.tile_classifier.dataset_generator import \
    DatasetGenerator, prune_image_rows_cols


def plot_sampled_regions(
        image_id,
        data_dir,
        tile_width=224,
        tile_height=224
) -> np.ndarray:
    """Returns pruned, downsampled slide with boxes indicating sampled tissue
    """
    data_gen = DatasetGenerator(data_dir=data_dir,
                                use_downsampled_slide=False,
                                tile_width=tile_width,
                                tile_height=tile_height,
                                fg_thresh=0.9)

    tiles = data_gen.get_tiles_for_slide(image_id=image_id)
    data_dir = Path(data_dir)
    with OpenSlide(data_dir / f'{image_id}.tif') as slide:
        downsampled_slide = data_gen.downsample_slide(slide=slide)
        downsample_factor = \
            data_gen.get_downsample_factor_for_slide(slide=slide)

    for tile in tiles:
        x, y = tile
        cv2.rectangle(downsampled_slide, pt1=(x, y),
                      pt2=(x + int(tile_width / downsample_factor),
                           y + int(tile_height / downsample_factor)),
                      color=(0, 255, 0), thickness=3)
    mask = data_gen.segment_slide(slide=downsampled_slide)
    pruned = prune_image_rows_cols(im=downsampled_slide, mask=mask)

    return pruned
