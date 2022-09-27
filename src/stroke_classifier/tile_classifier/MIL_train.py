import os
from typing import Tuple, Dict

import mlflow
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm

from stroke_classifier.tile_classifier.mil_dataset import get_dataloader

parser = argparse.ArgumentParser(
    description='MIL-nature-medicine-2019 tile classifier training script')
parser.add_argument('--train_dataset_path', type=str, required=True,
                    help='path to train dataset')
parser.add_argument('--val_dataset_path', type=str,
                    help='path to val dataset')
parser.add_argument('--output', type=str, default='.',
                    help='path to output file')
parser.add_argument('--batch_size', type=int, default=512,
                    help='mini-batch size (default: 512)')
parser.add_argument('--nepochs', type=int, default=100,
                    help='number of epochs')
parser.add_argument('--workers', default=0, type=int,
                    help='number of data loading workers')
parser.add_argument('--test_every', default=10, type=int,
                    help='test on val every (default: 10)')
parser.add_argument('--pos_train_loss_weight', default=0.5, type=float,
                    help='unbalanced positive class weight (default: 0.5, '
                         'balanced classes)')
parser.add_argument('--pos_evaluation_loss_weight', default=0.5, type=float,
                    help='Weight to use for the positive class when evaluating'
                         'performance')
parser.add_argument('--k', default=1, type=int,
                    help='top k tiles are assumed to be of the same class as '
                         'the slide (default: 1, standard MIL)')
parser.add_argument('--mlflow_tracking_uri')
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--mlflow_tag')
parser.add_argument('--early_stopping_patience', default=10, type=int)


def main():
    args = parser.parse_args()

    model = models.resnet34(True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    if torch.cuda.is_available():
        model.cuda()

    if args.pos_train_loss_weight == 0.5:
        weight = None
    else:
        weight = torch.Tensor([1-args.pos_train_loss_weight,
                          args.pos_train_loss_weight])
    criterion = nn.CrossEntropyLoss(weight=weight)
    if torch.cuda.is_available():
        criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                           weight_decay=args.weight_decay)

    cudnn.benchmark = True

    train_loader = get_dataloader(dataset_path=args.train_dataset_path,
                                  batch_size=args.batch_size,
                                  n_workers=args.workers,
                                  mode='train')
    train_inference_loader = get_dataloader(
        dataset_path=args.train_dataset_path,
        batch_size=args.batch_size,
        n_workers=args.workers,
        mode='inference')
    if args.val_dataset_path:
        val_loader = get_dataloader(dataset_path=args.val_dataset_path,
                                    batch_size=args.batch_size,
                                    n_workers=args.workers,
                                    mode='inference')
    else:
        val_loader = None

    track_using_mflow = args.mlflow_tracking_uri is not None
    if track_using_mflow:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment('mayo_clinic_stroke_kaggle')

        if args.mlflow_tag:
            mlflow.set_tag('notes', args.mlflow_tag)

        mlflow.log_params({
            'architecture': 'resnet34',
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'k': args.k,
            'pos_train_loss_weight': args.pos_train_loss_weight,
            'batch_size': args.batch_size,
            'early_stopping_patience': args.early_stopping_patience
        })

    train_slide_idxs = np.array(
        [slide_idx
         for _, _, _, slide_idx in train_inference_loader.dataset.tiles])
    if val_loader is not None:
        val_slide_idxs = np.array(
            [slide_idx
             for _, _, _, slide_idx in val_loader.dataset.tiles])
    else:
        val_slide_idxs = None

    print(f'Number of tiles in train: {len(train_loader.dataset.tiles)}')
    if val_loader is not None:
        print(f'Number of tiles in val: {len(val_loader.dataset.tiles)}')

    best_metric = float('inf')
    early_stopping_patience = args.early_stopping_patience
    time_since_best_epoch = 0
    best_epoch = 0

    for epoch in range(args.nepochs):
        print(f'Epoch {epoch+1}')
        print('===============')
        train_tile_probs, train_error = get_inference_for_epoch(
            data_loader=train_inference_loader,
            model=model,
            slide_idxs=train_slide_idxs,
            batch_size=args.batch_size,
            pos_evaluation_loss_weight=args.pos_evaluation_loss_weight
        )
        topk = get_topk_tiles(
            slide_indices=train_slide_idxs,
            tile_probs=train_tile_probs,
            k=args.k)
        train_loader.dataset.set_top_k_tiles(top_k_indices=topk)
        train_loss = train(train_loader, model, criterion, optimizer)
        print(f'Training\tEpoch: [{epoch+1}/{args.nepochs}]\t'
              f'Tile Loss: {train_loss}')
        if track_using_mflow:
            mlflow.log_metric(key='train_tile_loss', value=train_loss,
                              step=epoch)
            mlflow.log_metric(key='train_slide_loss',
                              value=train_error['log_loss'],
                              step=epoch)

        if val_loader is not None and (epoch + 1) % args.test_every == 0:
            _, val_error = get_inference_for_epoch(
                data_loader=val_loader,
                model=model,
                slide_idxs=val_slide_idxs,
                batch_size=args.batch_size,
                pos_evaluation_loss_weight=args.pos_evaluation_loss_weight
            )
            print(f'Validation\tEpoch: [{epoch+1}/{args.nepochs}]\t'
                  f'FPR: {val_error["fpr"]}\t'
                  f'FNR: {val_error["fnr"]}\t'
                  f'Log Loss: {val_error["log_loss"]}\t')

            if track_using_mflow:
                mlflow.log_metric(key='val_slide_loss',
                                  value=val_error['log_loss'],
                                  step=epoch)
                mlflow.log_metric(key='val_fpr', value=val_error['fpr'],
                                  step=epoch)
                mlflow.log_metric(key='val_fnr', value=val_error['fnr'],
                                  step=epoch)

            # Save best model
            err = val_error['log_loss']
            if err < best_metric:
                time_since_best_epoch = 0
                best_epoch = epoch
                best_metric = err
                obj = {
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'val_log_loss': best_metric,
                    'optimizer': optimizer.state_dict()
                }
                torch.save(obj, os.path.join(args.output,
                                             'checkpoint_best.pth'))
            else:
                time_since_best_epoch += 1
                if time_since_best_epoch > early_stopping_patience:
                    mlflow.set_tag('best_epoch', best_epoch)
                    print('Stopping due to early stopping')
                    return


def tile_inference(loader: DataLoader, model: nn.Module,
                   batch_size: int):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset), 2)
    with torch.no_grad():
        for i, input in tqdm(enumerate(loader), desc='tile inference',
                             total=len(loader)):
            input, _ = input
            if torch.cuda.is_available():
                input = input.cuda()
            output = F.softmax(model(input), dim=1)
            start_idx = i * batch_size
            end_idx = i * batch_size + input.size(0)
            probs[start_idx:end_idx] = output.detach().clone()

    return probs.cpu().numpy()


def train(loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.
    for i, (input, target) in tqdm(enumerate(loader), desc='train',
                                   total=len(loader)):
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*input.size(0)
    return running_loss/len(loader.dataset)


def calc_err(
        pred: np.ndarray,
        probs: np.ndarray,
        true: np.ndarray,
        pos_loss_weight=0.5
):
    pred = np.argmax(pred, axis=1)
    tp = ((pred == 1) & (true == 1)).sum()
    fp = ((pred == 1) & (true == 0)).sum()
    fn = ((pred == 0) & (true == 1)).sum()
    tn = ((pred == 0) & (true == 0)).sum()
    fpr = fp / (fp + tn)
    fnr = fn / (tp + fn)

    weighted_log_loss = calc_weighted_log_loss_kaggle(
        probs=probs,
        pos_weight=pos_loss_weight,
        target=true
    )
    return fpr, fnr, weighted_log_loss


def get_topk_tiles(slide_indices, tile_probs, k=1):
    """Gets top k tiles from each slide by probability of positive class"""
    order = np.lexsort((tile_probs[:, 1], slide_indices))
    slide_indices = slide_indices[order]
    index = np.empty(len(slide_indices), 'bool')
    index[-k:] = True
    index[:-k] = slide_indices[k:] != slide_indices[:-k]
    return list(order[index])


def slide_inference(
        slide_indices,
        tile_probs,
        classification_threshold=0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Gets classifications for each slide by treating tile with max
    probability as the slide-level prediction"""
    n_slides = len(np.unique(slide_indices))
    slide_probs = np.zeros((n_slides, 2))
    argmax_tiles = get_topk_tiles(
        slide_indices=slide_indices, tile_probs=tile_probs)
    slide_probs[slide_indices[argmax_tiles]] = tile_probs[argmax_tiles]
    pred = [x >= classification_threshold for x in slide_probs]
    pred = np.array(pred).astype(int)

    return slide_probs, pred


def calc_weighted_log_loss_kaggle(
        probs: np.ndarray,
        target: np.ndarray,
        pos_weight: float
):
    """Calculates weighted log loss where the average loss from each class
    is used in a weighted average"""
    log_probs = np.log(probs)
    weights = np.array([1-pos_weight, pos_weight])

    res = 0
    for c in (0, 1):
        class_log_probs = log_probs[target == c][:, c]
        class_weight = weights[c]
        res += class_weight * class_log_probs.mean()
    return - (res / weights.sum())


def get_inference_for_epoch(
        data_loader: DataLoader,
        model: nn.Module,
        batch_size: int,
        slide_idxs: np.ndarray,
        pos_evaluation_loss_weight=0.5
) -> Tuple[np.ndarray, Dict]:
    tile_probs = tile_inference(
        loader=data_loader,
        model=model,
        batch_size=batch_size)
    slide_probs, slide_pred = slide_inference(
        slide_indices=slide_idxs, tile_probs=tile_probs)
    fpr, fnr, log_loss = calc_err(
        pred=slide_pred,
        probs=slide_probs,
        true=np.array(
            [target for _, target in data_loader.dataset.slides]),
        pos_loss_weight=pos_evaluation_loss_weight
    )
    error = {
        'fpr': fpr,
        'fnr': fnr,
        'log_loss': log_loss
    }
    return tile_probs, error


if __name__ == '__main__':
    main()
