import os

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

from mil_dataset import get_dataloader

parser = argparse.ArgumentParser(
    description='MIL-nature-medicine-2019 tile classifier training script')
parser.add_argument('--train_dataset_path', type=str, default='',
                    help='path to train dataset saved with torch.save')
parser.add_argument('--val_dataset_path', type=str, default='',
                    help='path to val dataset saved with torch.save')
parser.add_argument('--output', type=str, default='.',
                    help='name of output file')
parser.add_argument('--batch_size', type=int, default=512,
                    help='mini-batch size (default: 512)')
parser.add_argument('--nepochs', type=int, default=100,
                    help='number of epochs')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--test_every', default=10, type=int,
                    help='test on val every (default: 10)')
parser.add_argument('--pos_loss_weight', default=0.5, type=float,
                    help='unbalanced positive class weight (default: 0.5, '
                         'balanced classes)')
parser.add_argument('--k', default=1, type=int,
                    help='top k tiles are assumed to be of the same class as '
                         'the slide (default: 1, standard MIL)')
parser.add_argument('--mlflow_tracking_uri', required=True)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)


def main():
    args = parser.parse_args()

    # cnn
    model = models.resnet34(True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.cuda()

    if args.weights == 0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1-args.pos_loss_weight, args.pos_loss_weight])
        criterion = nn.CrossEntropyLoss(w).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learing_rate,
                           weight_decay=args.weight_decay)

    cudnn.benchmark = True

    train_loader = get_dataloader(dataset_path=args.dataset_path,
                                  batch_size=args.batch_size,
                                  n_workers=args.workers,
                                  mode='train')
    if args.val_dataset_path:
        val_loader = get_dataloader(dataset_path=args.val_dataset_path,
                                    batch_size=args.batch_size,
                                    n_workers=args.workers,
                                    mode='inference')
    else:
        val_loader = None

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    train_slide_idxs = np.array(
        [x['slide'] for x in train_loader.dataset.tiles])
    val_slide_idxs = np.array([x['slide'] for x in val_loader.dataset.tiles])

    best_acc = 0

    # loop through epochs
    for epoch in range(args.nepochs):
        probs = tile_inference(run=epoch, loader=val_loader, model=model,
                               batch_size=args.batch_size,
                               n_epochs=args.nepochs)
        topk = get_topk_tiles(
            slide_indices=train_slide_idxs,
            tile_preds=probs,
            k=args.k)
        train_loader.dataset.get_top_k_tiles(top_k_indices=topk)
        loss = train(train_loader, model, criterion, optimizer)
        print(f'Training\tEpoch: [{epoch+1}/{args.nepochs}]\tLoss: {loss}')
        mlflow.log_metric(key='train_loss', value=loss, step=epoch)

        # Validation
        if val_loader is not None and (epoch + 1) % args.test_every == 0:
            probs = tile_inference(run=epoch, loader=val_loader, model=model,
                                   batch_size=args.batch_size,
                                   n_epochs=args.nepochs)
            pred = slide_inference(
                slide_indices=val_slide_idxs, tile_preds=probs)
            fpr, fnr, loss = calc_err(
                pred=pred,
                probs=probs,
                true=[x['target'] for x in val_loader.dataset.slides],
                criterion=criterion
            )
            print(f'Validation\tEpoch: [{epoch+1}/{args.nepochs}]\t'
                  f'FPR: {fpr}\t'
                  f'FNR: {fnr}\t'
                  f'Loss: {loss}')

            mlflow.log_metric(key='val_loss', value=loss, step=epoch)
            mlflow.log_metric(key='val_fpr', value=fpr, step=epoch)
            mlflow.log_metric(key='val_fnr', value=fnr, step=epoch)

            # Save best model
            err = (fpr + fnr) / 2.
            if 1-err >= best_acc:
                best_acc = 1-err
                obj = {
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict()
                }
                torch.save(obj, os.path.join(args.output,
                                             'checkpoint_best.pth'))


def tile_inference(run: int, loader: DataLoader, model: nn.Module,
                   batch_size: int, n_epochs: int):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, input in enumerate(loader):
            print(f'Inference\tEpoch: [{run + 1}/{n_epochs}]\t'
                  f'Batch: [{i + 1}/{len(loader)}]')
            input = input.cuda()
            output = F.softmax(model(input), dim=1)
            start_idx = i * batch_size
            end_idx = i * batch_size + input.size(0)
            probs[start_idx:end_idx] = output.detach()[:, 1].clone()
    return probs.cpu().numpy()


def train(loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.
    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*input.size(0)
    return running_loss/len(loader.dataset)


def calc_err(pred, probs, true, criterion: nn.CrossEntropyLoss):
    pred = np.array(pred)
    probs = np.array(probs)
    true = np.array(true)

    tp = (pred == 1 & true == 1).sum()
    fp = (pred == 1 & true == 0).sum()
    fn = (pred == 0 & true == 1).sum()
    tn = (pred == 0 & true == 0).sum()
    fpr = fp / (fp + tn)
    fnr = fn / (tp + fn)

    loss = criterion(probs, true)
    return fpr, fnr, loss


def get_topk_tiles(slide_indices, tile_preds, k=1):
    """Gets top k tiles from each slide by probability of positive class"""
    order = np.lexsort((tile_preds, slide_indices))
    slide_indices = slide_indices[order]
    index = np.empty(len(slide_indices), 'bool')
    index[-k:] = True
    index[:-k] = slide_indices[k:] != slide_indices[:-k]
    return list(order[index])


def slide_inference(
        slide_indices,
        tile_preds,
        classification_threshold=0.5
) -> np.ndarray:
    """Gets classifications for each slide by treating tile with max
    probability as the slide-level prediction"""
    n_slides = len(np.unique(slide_indices))
    slide_probs = np.zeros(n_slides)
    argmax_tiles = get_topk_tiles(
        slide_indices=slide_indices, tile_preds=tile_preds)
    slide_probs[slide_indices[argmax_tiles]] = tile_preds[argmax_tiles]
    pred = [x >= classification_threshold for x in slide_probs]
    pred = np.array(pred).astype(int)

    return pred


if __name__ == '__main__':
    main()
