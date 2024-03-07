from datetime import datetime

import models
from models import Resnet, ElderNet, ContrastiveLoss
from dataset.dataloader import SSL_dataset, subject_collate_simclr, worker_init_fn
from dataset.transformations import RotationAxisTimeSeries, RandomSwitchAxisTimeSeries
import utils
from utils import EarlyStopping, load_weights, set_seed

import os
import math
import numpy as np
import hydra

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


now = datetime.now()


def create_cosine_decay_with_warmup(optimizer, warmup_epochs, max_epochs):
    assert max_epochs > warmup_epochs, "max_epochs need to be greater than warmup_epochs"

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler


def evaluate_model(model, data_loader, device, cfg):
    model.eval()
    losses = []

    for batch_idx, (batch_data) in enumerate(data_loader):
        with torch.no_grad():
            # Extract the 2 views of the batch
            transform_1 = batch_data[0].to(device)
            transform_2 = batch_data[1].to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                hidden_features_transform_1 = model(transform_1.float().to(device))
                hidden_features_transform_2 = model(transform_2.float().to(device))

                loss_fn = ContrastiveLoss(batch_data.shape[1], device, temperature=cfg.model.temperature)
                loss = loss_fn(hidden_features_transform_1, hidden_features_transform_2)
                losses.append(loss.item())

    losses = np.array(losses)
    return losses


def log_performance(
        current_loss, writer, mode, epoch):
    # We want to have individual task performance
    # and an average loss performance
    # train_loss: numpy array
    # mode (str): train or test
    # overall = np.mean(np.mean(train_loss))
    loss = np.mean(current_loss)
    writer.add_scalar(mode + "/loss", loss, epoch)
    return loss


@hydra.main(config_path="conf", config_name="config_SimCLR", version_base='1.1')
def main(cfg):

    ####################
    #   Setting configurations
    ###################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed()
    num_epochs = cfg.model.num_epochs
    lr = cfg.model.lr
    batch_subject_num = cfg.dataloader.num_subjects  # number of subjects in each batch
    num_sample_per_subject = cfg.dataloader.num_samples  # number of acceleration windows per subject
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
    print("Number of epochs: %d" % num_epochs)
    print("Subjects per batch: %d" % batch_subject_num)
    print("Samples per subject: %d" % num_sample_per_subject)
    print("True batch size : %d" % true_batch_size)
    print("Tensor log dir: %s" % log_dir)

    ####################################
    # Set up model
    ####################################
    # Instantiate a network architecture
    if cfg.model.net == 'ElderNet':
        model = getattr(models, cfg.model.net)(main_trunk=Resnet(), cfg=cfg, is_simclr=True)
    else:
        model = getattr(models, cfg.model.net)(is_simclr=True)

    # Choose if to use an already pretrained model (such as the UKB model)
    if cfg.model.pretrained:
        # Use the model from the UKB paper
        if cfg.model.ssl_checkpoint_available:
            pretrained_model = utils.get_sslnet(pretrained=True)
            feature_extractor = pretrained_model.feature_extractor
            model = Resnet(feature_extractor=feature_extractor, is_simclr=True)
            if cfg.model.net == 'ElderNet':
                model = ElderNet(feature_extractor, cfg, is_simclr=True)
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
        collate_fn=subject_collate_simclr,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        num_workers=num_workers,
    )

    test_dataset = SSL_dataset(
        test_data_root,
        test_file_list_path,
        cfg,
        transform=my_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_subject_num,
        collate_fn=subject_collate_simclr,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        num_workers=num_workers,
    )

    ####################
    #   Set up Training
    ###################
    print("Start training")
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Define a lr scheduler with cosine decay with warm up
    scheduler = create_cosine_decay_with_warmup(optimizer, cfg.model.warm_up_step, num_epochs)

    # Training loop
    early_stopping = EarlyStopping(patience=cfg.model.patience, path=model_path, verbose=True)
    batch_size = len(train_loader)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    for epoch in range(num_epochs):
        model.train()
        accumulation_steps = cfg.model.accumulation_steps  # Gradient accumulation
        train_losses = []
        for batch_idx, (batch_data) in enumerate(train_loader):
            # Extract the 2 views of the batch
            transform_1 = batch_data[0]
            transform_2 = batch_data[1]

            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                torch.cuda.empty_cache()
                model.to(device)
                hidden_features_transform_1 = model(transform_1.float().to(device))
                hidden_features_transform_2 = model(transform_2.float().to(device))

                loss_fn = ContrastiveLoss(batch_data.shape[1], device, temperature=cfg.model.temperature)
                loss = loss_fn(hidden_features_transform_1, hidden_features_transform_2)
                # Scale the loss to account for gradient accumulation
                loss = loss / accumulation_steps
                train_losses.append(loss.cpu().detach().numpy())
                torch.cuda.empty_cache()

            if batch_idx % log_interval == 0:
                msg = "Train: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                    epoch + 1,
                    num_epochs,
                    batch_idx,
                    batch_size,
                    loss.item()
                )
                print(msg)
            scaler.scale(loss).backward()
            # Accumulate gradients for a fixed number of steps
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()

        # Update the learning rate
        scheduler.step()

        train_losses = np.array(train_losses)
        # Model evaluation
        test_losses = evaluate_model(model, test_loader, device, cfg)

        log_performance(train_losses,
                        writer,
                        "train",
                        epoch)

        test_loss = log_performance(test_losses,
                                    writer,
                                    "test",
                                    epoch)

        early_stopping(np.mean(test_loss), model)

        if early_stopping.early_stop:
            print("Early stopping")
            break


if __name__ == "__main__":
    main()
