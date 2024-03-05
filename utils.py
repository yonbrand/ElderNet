import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import copy

verbose = False
torch_cache_path = Path(__file__).parent / 'torch_hub_cache'
cuda = torch.cuda.is_available()

def set_seed(my_seed=0):
    random_seed = my_seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if cuda:
        torch.cuda.manual_seed_all(random_seed)

######################################################################################################################
# LOAD SSL
######################################################################################################################

def load_weights(weight_path, model, my_device="cpu"):
    pretrained_dict = torch.load(weight_path, map_location=my_device)
    pretrained_dict_v2 = copy.deepcopy(
        pretrained_dict
    )  # v2 has the right para names
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys such as the final linear layers
    #    we don't want linear layer weights either
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict_v2.items()
        if k in model_dict and k.split(".")[0] != "classifier"
    }

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    print("%d Weights loaded" % len(pretrained_dict))


def get_sslnet(tag='v1.0.0', pretrained=False):
    """
    Load and return the Self Supervised Learning (SSL) model from pytorch hub.

    :param str tag: Tag on the ssl-wearables repo to check out
    :param bool pretrained: Initialise the model with UKB self-supervised pretrained weights.
    :return: pytorch SSL model
    :rtype: nn.Module
    """

    repo_name = 'ssl-wearables'
    repo = f'OxWearables/{repo_name}:{tag}'

    if not torch_cache_path.exists():
        Path.mkdir(torch_cache_path, parents=True, exist_ok=True)

    torch.hub.set_dir(str(torch_cache_path))

    # find repo cache dir that matches repo name and tag
    cache_dirs = [f for f in torch_cache_path.iterdir() if f.is_dir()]
    repo_path = next((f for f in cache_dirs if repo_name in f.name and tag in f.name), None)

    if repo_path is None:
        repo_path = repo
        source = 'github'
    else:
        repo_path = str(repo_path)
        source = 'local'
        if verbose:
            print(f'Using local {repo_path}')

    sslnet: nn.Module = torch.hub.load(repo_path, 'harnet10', trust_repo=True, source=source, class_num=2,
                                       pretrained=pretrained, verbose=verbose)
    sslnet
    return sslnet


###################################################################################################################
# Early Stopping
###################################################################################################################
"""
Taken from https://github.com/Bjarten/early-stopping-pytorch
"""

class EarlyStopping:
    """Early stops the training if validation loss
    doesn't improve after a given patience."""

    def __init__(
        self,
        patience=5,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time v
                            alidation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each
                            validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity
                            to qualify as an improvement.
                            Default: 7
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            msg = "Validation loss decreased"
            msg = msg + f"({self.val_loss_min:.6f} --> {val_loss:.6f})"
            msg = msg + "Saving model ..."
            self.trace_func(msg)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss