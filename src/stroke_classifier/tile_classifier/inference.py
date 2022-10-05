from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm

from stroke_classifier.tile_classifier.mil_dataset import get_dataloader, \
    MILdataset


def tile_inference(loader: DataLoader, model: nn.Module,
                   batch_size: int):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset), 2)
    with torch.no_grad():
        for i, input in enumerate(tqdm(loader, desc='tile inference',
                             total=len(loader))):
            input, _ = input
            if torch.cuda.is_available():
                input = input.cuda()
            output = F.softmax(model(input), dim=1)
            start_idx = i * batch_size
            end_idx = i * batch_size + input.size(0)
            probs[start_idx:end_idx] = output.detach().clone()

    return probs.cpu().numpy()


def _get_slide_mean(dataset: MILdataset, tile_probs):
    """Gets mean score over all tiles in a slide for all classes
    """
    df = pd.DataFrame(tile_probs, columns=['CE', 'LAA'])
    df['slide'] = dataset.slide_idxs

    laa_mean = df.groupby('slide')['LAA'].mean()
    ce_mean = df.groupby('slide')['CE'].mean()
    res = pd.DataFrame({'CE': ce_mean, 'LAA': laa_mean})
    res = res.values
    res /= res.sum(axis=1).reshape(res.shape[0], 1)

    return res


def slide_inference(
        dataset: MILdataset,
        tile_probs
) -> Tuple[np.ndarray, np.ndarray]:
    """Gets classifications for each slide by choosing the max of tile probs
    for each class"""
    slide_probs = _get_slide_mean(
        dataset=dataset, tile_probs=tile_probs)
    pred = slide_probs.argmax(axis=1)

    return slide_probs, pred


def main():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--workers', default=0, type=int,
                        help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='mini-batch size (default: 512)')
    parser.add_argument('--output_dir', default='.')
    parser.add_argument('--dataset_meta',
                        required=True,
                        help='csv file mapping image_id to patient_id')
    args = parser.parse_args()

    model = models.resnet34(True)
    model.fc = nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    if torch.cuda.is_available():
        model.cuda()

    data_loader = get_dataloader(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        n_workers=args.workers,
        mode='inference'
    )

    tile_probs = tile_inference(
        loader=data_loader,
        model=model,
        batch_size=args.batch_size
    )

    slide_probs, _ = slide_inference(
        dataset=data_loader.dataset,
        tile_probs=tile_probs,
    )

    # Predictions are per patient_id, not image_id, so need to aggregate
    # predictions for a given patient_id
    slide_meta = pd.read_csv(args.dataset_meta)
    slide_probs = pd.DataFrame({
        'image_id': [image_id for _, image_id, _ in
                     data_loader.dataset.slides],
        'CE': slide_probs[:, 0],
        'LAA': slide_probs[:, 1]
    })
    slide_probs = slide_probs.merge(slide_meta, on='image_id')
    res = []
    for target in ('CE', 'LAA'):
        patient_mean = slide_probs.groupby('patient_id')[target].mean()
        res.append(patient_mean)
    res = pd.concat(res, axis=1)
    res /= res.sum(axis=1).values.reshape(res.shape[0], 1)
    res = res.round(6)

    out = Path(args.output_dir)
    res.to_csv(str(out / 'submission.csv'))


if __name__ == '__main__':
    main()
