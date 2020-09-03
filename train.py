import argparse
import datetime
import random
import json
import os
import time
from pathlib import Path

import apex
import cv2
import numpy as np
import pandas as pd
import torch
import torch.distributed as distributed
import torch.nn as nn
import torch.optim as optim
from apex import amp
from apex.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

import losses
import metrics
import models
from dataset import AlaskaDataset, load_dataset
from utils import AverageMeter, MetaData

cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)


def main(args):
    seed_everything(args.seed)

    model = models.build_model(encoder=args.encoder, pretrained=args.pretrained).cuda()
    if args.weights:
        model.load_state_dict(
            torch.load(args.weights, map_location=lambda storage, loc: storage)
        )

    if args.sync_bn:
        model = apex.parallel.convert_syncbn_model(model)

    optimizer = getattr(optim, args.optimizer)(
        model.parameters(), lr=args.learning_rate
    )

    model, optimizer = amp.initialize(
        model,
        optimizer,
        opt_level=args.opt_level,
        loss_scale=args.loss_scale,
        keep_batchnorm_fp32=args.keep_batchnorm_fp32,
    )

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.factor,
        patience=args.patience,
        min_lr=args.min_lr,
    )

    if args.distributed:
        model = DistributedDataParallel(model, delay_allreduce=True)

    train_loader, train_sampler, test_loader, test_sampler = data_loaders(args)
    criterion = getattr(losses, args.loss)()

    meta_data = MetaData()
    snapshots = Path(args.snapshots)
    color = "ycbcr" if args.ycbcr else "rgb"

    for epoch in range(args.start_epoch, args.end_epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train(train_loader, model, criterion, optimizer, epoch, args)

        loss, accuracy = validation(test_loader, model, criterion, args)

        lr_scheduler.step(loss)

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if args.local_rank == 0:
            print(
                "Validation: Time: {} | Epoch: {} | "
                "Accuracy: {:.4f} | Loss: {:.4f}".format(
                    current_time, epoch + 1, accuracy, loss
                )
            )

            if meta_data.accuracy < accuracy:
                state_dict = (
                    model.module.state_dict()
                    if args.distributed
                    else model.state_dict()
                )
                meta_data.update(state_dict, loss, accuracy, epoch + 1)
                torch.save(
                    meta_data.state_dict,
                    snapshots
                    / "{}_{}_{}_{}_{:.2e}.pth".format(
                        args.encoder, args.loss, color, args.size, args.learning_rate
                    ),
                )

    if args.local_rank == 0:
        torch.save(
            meta_data.state_dict,
            snapshots
            / "{}_{}_{}_{}_{:.2e}_{}.pth".format(
                args.encoder, args.loss, color, args.size, args.learning_rate, meta_data
            ),
        )


def train(data_loader, model, criterion, optimizer, epoch, args):
    model.train()

    loss_handler = AverageMeter()
    accuracy_handler = AverageMeter()

    if args.local_rank == 0:
        tq = tqdm(total=len(data_loader) * args.batch_size * args.world_size)

    for i, (image, target) in enumerate(data_loader):
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(image)  # .view(-1)

        loss = criterion(output, target)

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        if (i + 1) % args.step == 0:
            optimizer.step()
            optimizer.zero_grad()

        accuracy = metrics.accuracy(output, target)

        if args.distributed:
            loss = reduce_tensor(loss.data, args.world_size)
            accuracy = reduce_tensor(accuracy, args.world_size)

        loss_handler.update(loss)
        accuracy_handler.update(accuracy)

        if args.local_rank == 0:
            tq.set_description(
                "Epoch {}, lr {:.2e}".format(epoch + 1, get_learning_rate(optimizer))
            )
            tq.set_postfix(
                loss="{:.4f}".format(loss_handler.avg),
                accuracy="{:.4f}".format(accuracy_handler.avg),
            )
            tq.update(args.batch_size * args.world_size)

    if args.local_rank == 0:
        tq.close()


def validation(data_loader, model, criterion, args):
    model.eval()

    loss_handler = AverageMeter()
    accuracy_handler = AverageMeter()

    with torch.no_grad():
        for _, (image, target) in enumerate(data_loader):
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(image)  # .view(-1)

            loss = criterion(output, target)

            accuracy = metrics.accuracy(output, target)

            if args.distributed:
                loss = reduce_tensor(loss.data, args.world_size)
                accuracy = reduce_tensor(accuracy, args.world_size)

            loss_handler.update(loss)
            accuracy_handler.update(accuracy)

    return loss_handler.avg, accuracy_handler.avg


def data_loaders(args):
    train_data, test_data = load_dataset(args.data, test_size=args.test_size)

    if args.pseudo:
        pseudo = pd.read_csv(args.pseudo)
        train_data = train_data.append(pseudo)

    train_dataset = AlaskaDataset(
        train_data, train=True, ycbcr=args.ycbcr, size=args.size,
    )
    test_dataset = AlaskaDataset(
        test_data, train=False, ycbcr=args.ycbcr, size=args.size,
    )

    train_sampler = None
    test_sampler = None

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=test_sampler,
    )

    return train_loader, train_sampler, test_loader, test_sampler


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def reduce_tensor(tensor, size):
    reduced_tensor = tensor.clone()

    distributed.all_reduce(reduced_tensor, op=distributed.ReduceOp.SUM)
    return reduced_tensor / size


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train code")
    parser.add_argument("--encoder", type=str, default="resnet18")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--optimizer", type=str, default="Adam")

    parser.add_argument("--loss", type=str, default="binary_focal")
    parser.add_argument("--ycbcr", action="store_true")
    parser.add_argument("--size", type=int, default=512)

    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--end-epoch", type=int, default=5)

    parser.add_argument("--factor", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-6)

    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--sync-bn", action="store_true")
    parser.add_argument("--loss-scale", type=str, default="dynamic")
    parser.add_argument("--keep-batchnorm-fp32", type=str, default=None)

    parser.add_argument("--opt-level", type=str, default="O1")

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--pseudo", type=str, default=None)
    parser.add_argument("--test-size", type=float, default=0.01)
    parser.add_argument("--snapshots", type=str, default="snapshots")

    parser.add_argument("--seed", type=int, default=2020)
    args = parser.parse_args()

    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()

    main(args)
