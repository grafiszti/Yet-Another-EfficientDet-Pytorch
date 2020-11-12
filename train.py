# original author: signatrix
# adapted from https://github.com/signatrix/efficientdet/blob/master/train.py
# modified by Zylo117

import argparse
import datetime
import os
import traceback
from typing import Tuple

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.autonotebook import tqdm
import albumentations as albu
import cv2

from backbone import EfficientDetBackbone
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from efficientdet.loss import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import (
    replace_w_sync_bn,
    CustomDataParallel,
    get_last_weights,
    init_weights,
    boolean_string,
)
from consts import INVALID_STATE_DICT_LOAD_ERROR

from config import input_sizes


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_args():
    parser = argparse.ArgumentParser(
        "Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117"
    )
    parser.add_argument(
        "-p",
        "--project",
        type=str,
        default="projects/coco.yml",
        help="project file path that contains parameters",
    )
    parser.add_argument(
        "-c",
        "--compound_coef",
        type=int,
        default=0,
        help="coefficients of efficientdet",
    )
    parser.add_argument(
        "-n", "--num_workers", type=int, default=12, help="num_workers of dataloader"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=12,
        help="The number of images per batch among all devices",
    )
    parser.add_argument(
        "--head_only",
        type=boolean_string,
        default=False,
        help="whether finetunes only the regressor and the classifier, "
        "useful in early stage convergence or small/easy dataset",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw",
        help="select optimizer for training, "
        "suggest using 'admaw' until the"
        " very final stage then switch to 'sgd'",
    )
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Number of epoches between valing phases",
    )
    parser.add_argument(
        "--save_interval", type=int, default=500, help="Number of steps between saving"
    )
    parser.add_argument(
        "--es_min_delta",
        type=float,
        default=0.0,
        help="Early stopping's parameter: minimum change loss to qualify as an improvement",
    )
    parser.add_argument(
        "--es_patience",
        type=int,
        default=0,
        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.",
    )
    parser.add_argument(
        "--data_path", type=str, default="datasets/", help="the root folder of dataset"
    )
    parser.add_argument("--log_path", type=str, default="logs/")
    parser.add_argument(
        "-w",
        "--load_weights",
        type=str,
        default=None,
        help="whether to load weights from a checkpoint, set None to initialize, set 'last' to load last checkpoint",
    )
    parser.add_argument("--saved_path", type=str, default="logs/")
    parser.add_argument(
        "--debug",
        type=boolean_string,
        default=False,
        help="whether visualize the predicted boxes of training, "
        "the output images will be in test/",
    )

    args = parser.parse_args()
    return args


class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(
                classification,
                regression,
                anchors,
                annotations,
                imgs=imgs,
                obj_list=obj_list,
            )
        else:
            cls_loss, reg_loss = self.criterion(
                classification, regression, anchors, annotations
            )
        return cls_loss, reg_loss


def get_inference_transform(
    input_size: int, mean: Tuple[float, float, float], std: Tuple[float, float, float]
):
    return albu.Compose(
        [
            albu.Normalize(mean=mean, std=std),
            albu.Resize(width=input_size, height=input_size),
            albu.PadIfNeeded(
                min_width=input_size,
                min_height=input_size,
                border_mode=cv2.BORDER_CONSTANT,
            ),
        ]
    )


def get_inference_transform_weak(
    input_size: int, mean: Tuple[float, float, float], std: Tuple[float, float, float]
):
    return transforms.Compose([Normalizer(mean=mean, std=std), Resizer(input_size)])


def get_training_transform(
    input_size: int, mean: Tuple[float, float, float], std: Tuple[float, float, float]
):
    return albu.Compose(
        [
            albu.Normalize(mean=mean, std=std),
            albu.Flip(),
            albu.Blur(blur_limit=3, p=0.1),
            albu.ShiftScaleRotate(p=0.5),
            albu.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.2
            ),
            albu.Resize(width=input_size, height=input_size),
            albu.PadIfNeeded(min_width=input_size, min_height=input_size),
        ]
    )


def get_training_transform_weak(
    input_size: int, mean: Tuple[float, float, float], std: Tuple[float, float, float]
):
    return transforms.Compose(
        [Normalizer(mean=mean, std=std), Augmenter(), Resizer(input_size)]
    )


def run(opt):
    params = Params(opt.project)

    _set_gpus_number(params)
    _set_seed()
    _create_missing_dirs(opt, params)

    training_params = {
        "batch_size": opt.batch_size,
        "shuffle": True,
        "drop_last": True,
        "collate_fn": collater,
        "num_workers": opt.num_workers,
    }

    val_params = {
        "batch_size": opt.batch_size,
        "shuffle": False,
        "drop_last": True,
        "collate_fn": collater,
        "num_workers": opt.num_workers,
    }

    training_generator = DataLoader(
        CocoDataset(
            root_dir=os.path.join(opt.data_path, params.project_name),
            set=params.train_set,
            transform=get_training_transform(
                input_sizes[opt.compound_coef], params.mean, params.std
            ),
        ),
        **training_params,
    )

    val_generator = DataLoader(
        CocoDataset(
            root_dir=os.path.join(opt.data_path, params.project_name),
            set=params.val_set,
            transform=get_inference_transform(
                input_sizes[opt.compound_coef], params.mean, params.std
            ),
        ),
        **val_params,
    )

    model = EfficientDetBackbone(
        num_classes=len(params.obj_list),
        compound_coef=opt.compound_coef,
        ratios=eval(params.anchors_ratios),
        scales=eval(params.anchors_scales),
    )

    # load last weights
    if opt.load_weights is not None:
        if opt.load_weights.endswith(".pth"):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)

        try:
            last_step = int(os.path.basename(weights_path).split("_")[-1].split(".")[0])
        except:
            last_step = 0

        try:
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(INVALID_STATE_DICT_LOAD_ERROR.format(e))

        print(
            f"[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}"
        )
    else:
        last_step = 0
        print("[Info] initializing weights...")
        init_weights(model)

    # freeze backbone if train head_only
    if opt.head_only:
        _freeze_model_backbone(model)

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if params.num_gpus > 1 and opt.batch_size // params.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    writer = _create_summary_writer(opt.log_path)

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, debug=opt.debug)

    if params.num_gpus > 0:
        model = model.cuda()
        if params.num_gpus > 1:
            model = CustomDataParallel(model, params.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)

    optimizer = _get_optimizer(model, opt.optim, opt.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, verbose=True
    )

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    step = max(0, last_step)
    model.train()

    num_iter_per_epoch = len(training_generator)

    try:
        for epoch in range(opt.num_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            # training
            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iter, data in enumerate(progress_bar):
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:
                    imgs = data["img"]
                    annot = data["annot"]

                    if params.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    optimizer.zero_grad()
                    print("SHAAAAAAPEEEE???????????????????????")
                    print(imgs.shape)
                    print(annot.shape)
                    print("SHAAAAAAPEEEE!!!!!!!!!!!!!!!!!!!!!!!")

                    cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        "Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}".format(
                            step,
                            epoch,
                            opt.num_epochs,
                            iter + 1,
                            num_iter_per_epoch,
                            cls_loss.item(),
                            reg_loss.item(),
                            loss.item(),
                        )
                    )
                    writer.add_scalars("Loss", {"train": loss}, step)
                    writer.add_scalars("Regression_loss", {"train": reg_loss}, step)
                    writer.add_scalars("Classfication_loss", {"train": cls_loss}, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]["lr"]
                    writer.add_scalar("learning_rate", current_lr, step)

                    step += 1

                    if step % opt.save_interval == 0 and step > 0:
                        print("Step interval reached, saving model...")
                        _save_checkpoint(
                            model=model,
                            output_path=os.path.join(
                                opt.saved_path,
                                f"efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth",
                            ),
                        )

                except Exception as e:
                    print("[Error]", traceback.format_exc())
                    print(e)
                    continue
            scheduler.step(np.mean(epoch_loss))

            # validation
            if epoch % opt.val_interval == 0:
                model.eval()
                loss_regression_ls = []
                loss_classification_ls = []
                for iter, data in enumerate(val_generator):
                    with torch.no_grad():
                        imgs = data["img"]
                        annot = data["annot"]

                        if params.num_gpus == 1:
                            imgs = imgs.cuda()
                            annot = annot.cuda()

                        cls_loss, reg_loss = model(
                            imgs, annot, obj_list=params.obj_list
                        )
                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()

                        loss = cls_loss + reg_loss
                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        loss_classification_ls.append(cls_loss.item())
                        loss_regression_ls.append(reg_loss.item())

                cls_loss = np.mean(loss_classification_ls)
                reg_loss = np.mean(loss_regression_ls)
                loss = cls_loss + reg_loss

                print(
                    "Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}".format(
                        epoch, opt.num_epochs, cls_loss, reg_loss, loss
                    )
                )
                writer.add_scalars("Loss", {"val": loss}, step)
                writer.add_scalars("Regression_loss", {"val": reg_loss}, step)
                writer.add_scalars("Classfication_loss", {"val": cls_loss}, step)

                if loss + opt.es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch

                    print("Saving best loss model...")
                    _save_checkpoint(
                        model=model,
                        output_path=os.path.join(
                            opt.saved_path,
                            f"efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth",
                        ),
                    )

                model.train()

                # Early stopping
                if epoch - best_epoch > opt.es_patience > 0:
                    print(
                        "[Info] Stop training at epoch {}. The lowest loss achieved is {}".format(
                            epoch, best_loss
                        )
                    )
                    break
    except KeyboardInterrupt:
        print("Keyboard interrupt occured, saving model ...")
        _save_checkpoint(
            model=model,
            output_path=os.path.join(
                opt.saved_path, f"efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth"
            ),
        )
        writer.close()
    writer.close()


def _get_optimizer(model: ModelWithLoss, optim: str, lr: float):
    if optim == "adamw":
        return torch.optim.AdamW(model.parameters(), lr)
    return torch.optim.SGD(model.parameters(), lr, momentum=0.9, nesterov=True)


def _create_summary_writer(log_path: str):
    return SummaryWriter(
        log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/'
    )


def _freeze_model_backbone(model: EfficientDetBackbone):
    def freeze_backbone(m):
        classname = m.__class__.__name__
        for ntl in ["EfficientNet", "BiFPN"]:
            if ntl in classname:
                for param in m.parameters():
                    param.requires_grad = False

    model.apply(freeze_backbone)
    print("[Info] freezed backbone")


def _create_missing_dirs(opt, params):
    opt.saved_path = opt.saved_path + f"/{params.project_name}/"
    opt.log_path = opt.log_path + f"/{params.project_name}/tensorboard/"
    os.makedirs(opt.saved_path, exist_ok=True)
    os.makedirs(opt.log_path, exist_ok=True)


def _set_seed():
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)


def _set_gpus_number(params):
    if params.num_gpus == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def _save_checkpoint(model: ModelWithLoss, output_path: str):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), output_path)
    else:
        torch.save(model.model.state_dict(), output_path)
    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    opt = get_args()
    run(opt)
