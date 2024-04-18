import os
from models import Resnet, ElderNet
import torch
import copy

dependencies = ["torch"]

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

def eldernet_ssl(pretrained=False, my_device="cpu"):
    """
    load the SSL pretrained ElderNet (without the gait detection fine-tuning)

    Input:
    pretrained (bool) : load ElderNet, pretrained on the MAP dataset

    Output:
    model : the ElderNet model

    Example:
    repo = 'yonbrand/ElderNet'
    model = torch.hub.load(repo, 'eldernet_ssl',pretrained=True)
    x = np.random.rand(1, 3, 300)
    x = torch.FloatTensor(x)
    model(x)

    """
    # Call the model, load pretrained weights
    feature_extractor = Resnet().feature_extractor
    model = ElderNet(feature_extractor, is_eva=True)
    if pretrained:
        dirname = os.path.dirname(__file__)
        checkpoint = os.path.join(dirname, "", "model_check_point", "ssl_weights.mdl")
        load_weights(checkpoint, model, my_device)

    return model


def eldernet_ft(my_device="cpu"):
    """
    load  ElderNet fine-tuned on the Mobilise-D data for gait detection

    Output:
    model : the fine-tuned model

    Example:
    repo = 'yonbrand/ElderNet'
    model = torch.hub.load(repo, 'eldernet_ft',pretrained=True)
    x = np.random.rand(1, 3, 300)
    x = torch.FloatTensor(x)
    model(x)

    """
    feature_extractor = Resnet().feature_extractor
    model = ElderNet(feature_extractor, is_eva=True)
    dirname = os.path.dirname(__file__)
    checkpoint = os.path.join(dirname, "", "model_check_point", "ft_weights.pt")
    load_weights(checkpoint, model, my_device)
    return(model)