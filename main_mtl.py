import os
import numpy as np
import hydra


import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import models
from models import Resnet, ElderNet
from dataset.dataloader import SSL_dataset, subject_collate_mtl, worker_init_fn
from dataset.transformations import RotationAxisTimeSeries, RandomSwitchAxisTimeSeries
import utils
from utils import EarlyStopping, load_weights, set_seed

from datetime import datetime

now = datetime.now()

""""
This code is mainly taken from here: https://github.com/OxWearables/ssl-wearables
Muti-tasking learning for self-supervised wearable model
"""


################################
#
#
#       helper functions
#
#
################################
def set_up_data4train(my_X, aot_y, scale_y, permute_y, time_w_y, device):
    my_X = my_X.to(device, dtype=torch.float16)
    aot_y = aot_y.to(device, dtype=torch.long)
    scale_y = scale_y.to(device, dtype=torch.long)
    permute_y = permute_y.to(device, dtype=torch.long)
    time_w_y = time_w_y.to(device, dtype=torch.long)
    return my_X, aot_y, scale_y, permute_y, time_w_y


def evaluate_model(model, data_loader, device, cfg):
    model.eval()
    losses = []
    acces = []
    task_losses = []

    for i, (my_X, aot_y, scale_y, permute_y, time_w_y) in enumerate(
            data_loader
    ):
        with torch.no_grad():
            my_X, aot_y, scale_y, permute_y, time_w_y = set_up_data4train(
                my_X, aot_y, scale_y, permute_y, time_w_y, device
            )

            aot_y_pred, scale_y_pred, permute_y_pred, time_w_h_pred = model(
                my_X
            )

            loss, acc, task_loss = compute_loss(
                cfg,
                aot_y,
                scale_y,
                permute_y,
                time_w_y,
                aot_y_pred,
                scale_y_pred,
                permute_y_pred,
                time_w_h_pred,
            )
            losses.append(loss.item())
            acces.append(acc.item())
            task_losses.append(task_loss)
    losses = np.array(losses)
    acces = np.array(acces)
    task_losses = np.array(task_losses)
    return losses, acces, task_losses


def log_performance(
        current_loss, current_acces, writer, mode, epoch, task_name, task_loss=[]
):
    # We want to have individual task performance
    # and an average loss performance
    # train_loss: numpy array
    # mode (str): train or test
    # overall = np.mean(np.mean(train_loss))
    # rotataion_loss = np.mean(train_loss[:, ROTATION_IDX])
    # task_loss: is only true for all task config
    loss = np.mean(current_loss)
    acc = np.mean(current_acces)

    writer.add_scalar(mode + "/" + task_name + "_loss", loss, epoch)
    writer.add_scalar(mode + "/" + task_name + "_acc", acc, epoch)

    if len(task_loss) > 0:
        aot_loss = np.mean(task_loss[:, 0])
        permute_loss = np.mean(task_loss[:, 1])
        scale_loss = np.mean(task_loss[:, 2])
        time_w_loss = np.mean(task_loss[:, 3])
        writer.add_scalar(mode + "/aot_loss", aot_loss, epoch)
        writer.add_scalar(mode + "/permute_loss", permute_loss, epoch)
        writer.add_scalar(mode + "/scale_loss", scale_loss, epoch)
        writer.add_scalar(mode + "/time_w_loss", time_w_loss, epoch)

    return loss

def set_linear_scale_lr(model, cfg):
    """Allow for large minibatch
    https://arxiv.org/abs/1706.02677
    1. Linear scale learning rate in proportion to minibatch size
    2. Linear learning scheduler to allow for warm up for the first 5 epochs
    """
    # reference batch size and learning rate
    # lr: 0.0001 batch_size: 512

    reference_lr = 0.0001
    ref_batch_size = 512.0
    optimizer = optim.Adam(
        model.parameters(), lr=reference_lr, amsgrad=True
    )
    k = (
                1.0
                * cfg.dataloader.num_samples
                * cfg.dataloader.num_subjects
        ) / ref_batch_size
    scale_ratio = k ** (1.0 / 5.0)
    # linear warm up to account for large batch size
    lambda1 = lambda epoch: scale_ratio ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    return optimizer, scheduler


def compute_acc(logits, true_y):
    pred_y = torch.argmax(logits, dim=1)
    acc = torch.sum(pred_y == true_y)
    acc = 1.0 * acc / (pred_y.size()[0])
    return acc


def compute_loss(
        cfg,
        aot_y,
        scale_y,
        permute_y,
        time_w_y,
        aot_y_pred,
        scale_y_pred,
        permute_y_pred,
        time_w_h_pred,
):
    entropy_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0
    total_task = 0
    total_acc = 0
    aot_loss = 0
    permute_loss = 0
    scale_loss = 0
    time_w_loss = 0


    if cfg.task.time_reversal:
        aot_loss = entropy_loss_fn(aot_y_pred, aot_y)
        total_loss += aot_loss
        total_acc += compute_acc(aot_y_pred, aot_y)
        total_task += 1
        aot_loss = aot_loss.item()

    if cfg.task.permutation:
        permute_loss = entropy_loss_fn(permute_y_pred, permute_y)
        total_loss += permute_loss
        total_acc += compute_acc(permute_y_pred, permute_y)
        total_task += 1
        permute_loss = permute_loss.item()

    if cfg.task.scale:
        scale_loss = entropy_loss_fn(scale_y_pred, scale_y)
        total_loss += scale_loss
        total_acc += compute_acc(scale_y_pred, scale_y)
        total_task += 1
        scale_loss = scale_loss.item()

    if cfg.task.time_warped:
        time_w_loss = entropy_loss_fn(time_w_h_pred, time_w_y)
        total_loss += time_w_loss
        total_acc += compute_acc(time_w_h_pred, time_w_y)
        total_task += 1
        time_w_loss = time_w_loss.item()

    return (
        total_loss / total_task,
        total_acc / total_task,
        [aot_loss, permute_loss, scale_loss, time_w_loss],
    )


@hydra.main(config_path="conf", config_name="config_mtl", version_base='1.1')
def main(cfg):
    ####################
    #   Setting configurations
    ###################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed()
    num_epochs = cfg.model.num_epochs
    lr = cfg.model.lr
    batch_subject_num = cfg.dataloader.num_subjects #number of subjects in each batch
    num_sample_per_subject = cfg.dataloader.num_samples #number of acceleration windows per subject
    true_batch_size = batch_subject_num * num_sample_per_subject

    # data config
    train_data_root = cfg.data.train_data_root
    test_data_root = cfg.data.test_data_root
    train_file_list_path = cfg.data.train_file_list
    test_file_list_path = cfg.data.test_file_list
    log_interval = cfg.data.log_interval

    main_log_dir = cfg.data.log_path
    dt_string = now.strftime("%d-%m-%Y_%H_%M_%S")
    run_name = cfg.model.name + "_" + dt_string
    log_dir = os.path.join(main_log_dir, run_name)
    general_model_path = os.path.join(
        main_log_dir,
        "models",
        run_name
    )
    model_path = general_model_path + ".mdl"

    print("Model name: %s" % cfg.model.name)
    print("Learning rate: %f" % lr)
    print("Number of epoches: %d" % num_epochs)
    print("Subjects per batch: %d" % batch_subject_num)
    print("Samples per subject: %d" % num_sample_per_subject)
    print("True batch size : %d" % true_batch_size)
    print("Tensor log dir: %s" % log_dir)

    ####################################
    # Set up model
    ####################################
    # Instantiate a network architecture
    if cfg.model.net == 'ElderNet':
        feature_extractor = Resnet().feature_extractor
        model = getattr(models, cfg.model.net)(feature_extractor, head=cfg.model.head,
                                               non_linearity=cfg.model.non_linearity, is_eva=True)
    else:
        model = getattr(models, cfg.model.net)(is_mtl=True)

    # Choose if to use an already pretrained model (such as the UKB model)
    if cfg.model.pretrained:
        # Use the model from the UKB paper
        if not cfg.model.ssl_checkpoint_available:
            pretrained_model = utils.get_sslnet(pretrained=True)
            feature_extractor = pretrained_model.feature_extractor
            model = Resnet(feature_extractor=feature_extractor, is_mtl=True)
            if cfg.model.net == 'ElderNet':
                model = ElderNet(feature_extractor, cfg, is_mtl=True)
        # Use a pretrained model of your own
        else:
            load_weights(cfg.model.trained_model_path, model, device)

    num_workers = 1
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("Num of params %d " % pytorch_total_params)

    ####################
    #   Set up data
    ###################
    writer = SummaryWriter(log_dir)

    augmentations = []
    if cfg.augmentation.axis_switch:
        augmentations.append(RandomSwitchAxisTimeSeries())

    if cfg.augmentation.rotation:
        augmentations.append(RotationAxisTimeSeries())

    my_transform = transforms.Compose(augmentations) if augmentations else None

    # Create datasets from the acceleration signals to train and test the model
    train_dataset = SSL_dataset(
        train_data_root,
        train_file_list_path,
        cfg,
        transform=my_transform
    )

    train_sampler = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_subject_num,
        collate_fn=subject_collate_mtl,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        num_workers=num_workers,
    )

    test_dataset = SSL_dataset(
        test_data_root, test_file_list_path, cfg
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_subject_num,
        collate_fn=subject_collate_mtl,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        num_workers=num_workers,
    )


    ####################
    #   Set up Training
    ###################
    optimizer, scheduler = set_linear_scale_lr(model, cfg)
    total_step = len(train_loader)

    print("Start training")
    # Create an early stop object to prevent over-fitting
    early_stopping = EarlyStopping(
        patience=cfg.model.patience, path=model_path, verbose=True
    )
    accumulation_steps = cfg.model.accumulation_steps # Gradient accumulation
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    for epoch in range(num_epochs):

        model.train()

        train_losses = []
        train_acces = []
        task_losses = []

        for i, (my_X, aot_y, scale_y, permute_y, time_w_y) in enumerate(
                train_loader
        ):
            # the labels for all tasks are always generated
            my_X, aot_y, scale_y, permute_y, time_w_y = set_up_data4train(
                my_X, aot_y, scale_y, permute_y, time_w_y, device)
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):

                model = model.to(device)
                aot_y_pred, scale_y_pred, permute_y_pred, time_w_h_pred = model(
                    my_X
                )
                loss, acc, task_loss = compute_loss(
                    cfg,
                    aot_y,
                    scale_y,
                    permute_y,
                    time_w_y,
                    aot_y_pred,
                    scale_y_pred,
                    permute_y_pred,
                    time_w_h_pred,
                )

            scaler.scale(loss).backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1 == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
            # loss.backward()
            # optimizer.step()

            if i % log_interval == 0:
                msg = "Train: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, ACC : {:.4f}".format(
                    epoch + 1,
                    num_epochs,
                    i,
                    total_step,
                    loss.item(),
                    acc.item(),
                )
                print(msg)
            train_losses.append(loss.cpu().detach().numpy())
            train_acces.append(acc.cpu().detach().numpy())
            task_losses.append(task_loss)

        train_task_losses = np.array(task_losses)
        if epoch < cfg.model.warm_up_step:
            scheduler.step()

        train_losses = np.array(train_losses)
        train_acces = np.array(train_acces)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            test_losses, test_acces, task_losses = evaluate_model(model, test_loader, device, cfg)

        # logging
        log_performance(
            train_losses,
            train_acces,
            writer,
            "train",
            epoch,
            cfg.task.task_name,
            task_loss=train_task_losses,
        )

        test_loss = log_performance(
            test_losses,
            test_acces,
            writer,
            "test",
            epoch,
            cfg.task.task_name,
            task_loss=task_losses,
        )

        early_stopping(test_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break


if __name__ == "__main__":
    main()
