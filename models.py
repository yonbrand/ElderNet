import  numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


####################################################################################################################
# Contrastive Loss
####################################################################################################################
# This implementation is based on the code of  Nikolas Adaloglou: https://theaisummer.com/simclr/

class ContrastiveLoss(nn.Module):
   """
   Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
   """
   def __init__(self, batch_size, device, temperature=0.5):
       super().__init__()
       self.batch_size = batch_size
       self.device = device
       self.temperature = temperature
       self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

   def calc_similarity_batch(self, a, b):
       representations = torch.cat([a, b], dim=0)
       similarity_fn = nn.CosineSimilarity(dim=2)
       similarity_matrix = similarity_fn(representations.unsqueeze(1), representations.unsqueeze(0))
       return similarity_matrix

   def forward(self, proj_1, proj_2):
       """
       proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
       where corresponding indices are pairs
       z_i, z_j in the SimCLR paper
       """

       z_i = F.normalize(proj_1, p=2, dim=1)
       z_j = F.normalize(proj_2, p=2, dim=1)

       similarity_matrix = self.calc_similarity_batch(z_i, z_j)

       sim_ij = torch.diag(similarity_matrix, self.batch_size)
       sim_ji = torch.diag(similarity_matrix, -self.batch_size)

       positives = torch.cat([sim_ij, sim_ji], dim=0)

       nominator = torch.exp(positives / self.temperature)

       denominator = self.mask.to(self.device) * torch.exp(similarity_matrix / self.temperature)

       all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
       loss = torch.sum(all_losses) / (2 * self.batch_size)
       return loss

####################################################################################################################


class Classifier(nn.Module):
    def __init__(self, input_size=1024, output_size=2):
        super(Classifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        y_pred = self.linear1(x)
        return y_pred


class SigmoidClassifier(nn.Module):
    def __init__(self, input_size=1024, output_size=1):
        super(SigmoidClassifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y_pred = self.sigmoid(self.linear1(x))
        return y_pred


class EvaClassifier(nn.Module):
    def __init__(self, input_size=1024, nn_size=512, output_size=2):
        super(EvaClassifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, nn_size)
        self.linear2 = torch.nn.Linear(nn_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


class LinearLayers(nn.Module):
    """
    Additional layers to tailor ElderNet for the MAP data
    """
    def __init__(self, input_size=1024, output_size=50, non_linearity=False):
        super(LinearLayers, self).__init__()
        assert input_size / 4 > 0 , "input size too small"
        assert  output_size <= (input_size/4), "output size needs to be smaller the input size/4"
        self.linear1 = torch.nn.Linear(input_size, int(input_size/2))
        self.linear2 = torch.nn.Linear(int(input_size/2), int(input_size/4))
        self.linear3 = torch.nn.Linear(int(input_size/4), output_size)
        self.relu = nn.ReLU()
        self.non_linearity = non_linearity
        weight_init(self)

    def forward(self, x):
        if self.non_linearity:
            fc1 = self.linear1(x)
            fc2 = self.linear2(self.relu(fc1))
            out = self.linear3(self.relu(fc2))
        else:
            fc1 = self.linear1(x)
            fc2 = self.linear2(fc1)
            out = self.linear3(fc2)
        return out


class Downsample(nn.Module):
    r"""Downsampling layer that applies anti-aliasing filters.
    For example, order=0 corresponds to a box filter (or average downsampling
    -- this is the same as AvgPool in Pytorch), order=1 to a triangle filter
    (or linear downsampling), order=2 to cubic downsampling, and so on.
    See https://richzhang.github.io/antialiased-cnns/ for more details.
    """

    def __init__(self, channels=None, factor=2, order=1):
        super(Downsample, self).__init__()
        assert factor > 1, "Downsampling factor must be > 1"
        self.stride = factor
        self.channels = channels
        self.order = order

        # Figure out padding and check params make sense
        # The padding is given by order*(factor-1)/2
        # so order*(factor-1) must be divisible by 2
        total_padding = order * (factor - 1)
        assert total_padding % 2 == 0, (
            "Misspecified downsampling parameters."
            "Downsampling factor and order must be such "
            "that order*(factor-1) is divisible by 2"
        )
        self.padding = int(order * (factor - 1) / 2)

        box_kernel = np.ones(factor)
        kernel = np.ones(factor)
        for _ in range(order):
            kernel = np.convolve(kernel, box_kernel)
        kernel /= np.sum(kernel)
        kernel = torch.Tensor(kernel)
        self.register_buffer(
            "kernel", kernel[None, None, :].repeat((channels, 1, 1))
        )

    def forward(self, x):
        return F.conv1d(
            x,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.shape[1],
        )


class ResBlock(nn.Module):
    r""" Basic bulding block in Resnets:

       bn-relu-conv-bn-relu-conv
      /                         \
    x --------------------------(+)->

    """

    def __init__(
        self, in_channels, out_channels, kernel_size=5, stride=1, padding=2
    ):

        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(x))
        x = self.conv1(x)
        x = self.relu(self.bn2(x))
        x = self.conv2(x)

        x = x + identity

        return x


class Resnet(nn.Module):
    r"""The general form of the architecture can be described as follows:

    x->[Conv-[ResBlock]^m-BN-ReLU-Down]^n->y

    In other words:

            bn-relu-conv-bn-relu-conv                        bn-
           /                         \                      /
    x->conv --------------------------(+)-bn-relu-down-> conv ----

    """

    def __init__(
        self,
        output_size=2,
        n_channels=3,
        resnet_version=1,
        epoch_len=10,
        feature_extractor=nn.Sequential(),
        is_mtl=False,
        is_simclr=False,
        is_eva=False
    ):
        super(Resnet, self).__init__()

        # Architecture definition. Each tuple defines
        # a basic Resnet layer Conv-[ResBlock]^m]-BN-ReLU-Down
        # isEva: change the classifier to two FC with ReLu
        # For example, (64, 5, 1, 5, 3, 1) means:
        # - 64 convolution filters
        # - kernel size of 5
        # - 1 residual block (ResBlock)
        # - ResBlock's kernel size of 5
        # - downsampling factor of 3
        # - downsampling filter order of 1
        # In the below, note that 3*3*5*5*4 = 900 (input size)
        if resnet_version == 1:
            if epoch_len == 5:
                cgf = [
                    (64, 5, 2, 5, 2, 2),
                    (128, 5, 2, 5, 2, 2),
                    (256, 5, 2, 5, 3, 1),
                    (256, 5, 2, 5, 3, 1),
                    (512, 5, 0, 5, 3, 1),
                ]
            elif epoch_len == 10:
                cgf = [
                    (64, 5, 2, 5, 2, 2),
                    (128, 5, 2, 5, 2, 2),
                    (256, 5, 2, 5, 5, 1),
                    (512, 5, 2, 5, 5, 1),
                    (1024, 5, 0, 5, 3, 1),
                ]
            else:
                cgf = [
                    (64, 5, 2, 5, 3, 1),
                    (128, 5, 2, 5, 3, 1),
                    (256, 5, 2, 5, 5, 1),
                    (512, 5, 2, 5, 5, 1),
                    (1024, 5, 0, 5, 4, 0),
                ]
        else:
            cgf = [
                (64, 5, 2, 5, 3, 1),
                (64, 5, 2, 5, 3, 1),
                (128, 5, 2, 5, 5, 1),
                (128, 5, 2, 5, 5, 1),
                (256, 5, 2, 5, 4, 0),
            ]  # smaller resnet
        in_channels = n_channels
        self.feature_extractor = feature_extractor
        for i, layer_params in enumerate(cgf):
            (
                out_channels,
                conv_kernel_size,
                n_resblocks,
                resblock_kernel_size,
                downfactor,
                downorder,
            ) = layer_params
            self.feature_extractor.add_module(
                f"layer{i+1}",
                Resnet.make_layer(
                    in_channels,
                    out_channels,
                    conv_kernel_size,
                    n_resblocks,
                    resblock_kernel_size,
                    downfactor,
                    downorder,
                ),
            )
            in_channels = out_channels


        self.is_mtl = is_mtl
        self.is_simclr = is_simclr
        self.is_eva = is_eva

        if self.is_mtl:
            self.aot_h = Classifier(
                input_size=out_channels, output_size=output_size
            )
            self.scale_h = Classifier(
                input_size=out_channels, output_size=output_size
            )
            self.permute_h = Classifier(
                input_size=out_channels, output_size=output_size
            )
            self.time_w_h = Classifier(
                input_size=out_channels, output_size=output_size
            )

        elif self.is_simclr:
            self.classifier = Classifier(input_size=out_channels, output_size=50)

        elif self.is_eva:
            self.classifier = EvaClassifier(
                input_size=out_channels, output_size=output_size
            )

        weight_init(self)

    @staticmethod
    def make_layer(
        in_channels,
        out_channels,
        conv_kernel_size,
        n_resblocks,
        resblock_kernel_size,
        downfactor,
        downorder=1,
    ):
        r""" Basic layer in Resnets:

        x->[Conv-[ResBlock]^m-BN-ReLU-Down]->

        In other words:

                bn-relu-conv-bn-relu-conv
               /                         \
        x->conv --------------------------(+)-bn-relu-down->

        """

        # Check kernel sizes make sense (only odd numbers are supported)
        assert (
            conv_kernel_size % 2
        ), "Only odd number for conv_kernel_size supported"
        assert (
            resblock_kernel_size % 2
        ), "Only odd number for resblock_kernel_size supported"

        # Figure out correct paddings
        conv_padding = int((conv_kernel_size - 1) / 2)
        resblock_padding = int((resblock_kernel_size - 1) / 2)

        modules = [
            nn.Conv1d(
                in_channels,
                out_channels,
                conv_kernel_size,
                1,
                conv_padding,
                bias=False,
                padding_mode="circular",
            )
        ]

        for i in range(n_resblocks):
            modules.append(
                ResBlock(
                    out_channels,
                    out_channels,
                    resblock_kernel_size,
                    1,
                    resblock_padding,
                )
            )

        modules.append(nn.BatchNorm1d(out_channels))
        modules.append(nn.ReLU(True))
        modules.append(Downsample(out_channels, downfactor, downorder))

        return nn.Sequential(*modules)

    def forward(self, x):
        feats = self.feature_extractor(x)
        if self.is_mtl:
            aot_y = self.aot_h(feats.view(x.shape[0], -1))
            scale_y = self.scale_h(feats.view(x.shape[0], -1))
            permute_y = self.permute_h(feats.view(x.shape[0], -1))
            time_w_h = self.time_w_h(feats.view(x.shape[0], -1))
            return aot_y, scale_y, permute_y, time_w_h
        elif self.is_simclr:
            return self.classifier(feats.view(x.shape[0], -1))
        elif self.is_eva:
            return self.classifier(feats.view(x.shape[0], -1))


class Unet(nn.Module):
    def __init__(self, as_head=False, is_mtl=False, is_simclr=False, is_eva=False):
        super().__init__()
        self.as_head = as_head #indicate if the Unet model used alone or as a head of the Resnet model
        self.is_mtl = is_mtl
        self.is_simclr = is_simclr
        self.is_eva = is_eva

        self.conv1_1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=8)
        self.conv1_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8)

        self.batchNormconv1 = nn.BatchNorm1d(64)

        self.conv2_1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8)
        self.conv2_2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8)
        self.batchNormconv2 = nn.BatchNorm1d(128)

        self.conv3_1 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=8)
        self.conv3_2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=8)
        self.batchNormconv3 = nn.BatchNorm1d(256)

        self.convTranspose1 = nn.ConvTranspose1d(256, 128, 2, stride=2)
        self.conv2_5 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=8)
        self.conv2_6 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8)
        self.batchNormconv4 = nn.BatchNorm1d(128)

        self.convTranspose2 = nn.ConvTranspose1d(128, 64, 2, stride=2)
        self.conv1_4 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=8)
        self.conv1_4_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8)
        self.conv1_5 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=1)
        self.conv1_6 = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1)
        self.fc = nn.Linear(300,50)

        if self.as_head:
            self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=8)
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 128)
            self.fc3 = nn.Linear(128, 50)

        self.drop1 = nn.Dropout1d(p=0.2)
        self.drop2 = nn.Dropout1d(p=0.5)

        if self.is_mtl and not self.as_head:
            self.aot_h = Classifier(
                input_size=50, output_size=2
            )
            self.scale_h = Classifier(
                input_size=50, output_size=2
            )
            self.permute_h = Classifier(
                input_size=50, output_size=2
            )
            self.time_w_h = Classifier(
                input_size=50, output_size=2
            )

        # Indicate if we are in evaluation mode (and then add a classification head)
        if self.is_eva:
            self.classifier = Classifier(50, 2)

        weight_init(self)

    def forward(self, x):
        pad_x = nn.ReflectionPad1d((3, 4))(x)
        conv1_1 = F.relu(self.conv1_1(pad_x))
        conv1_1 = nn.ReflectionPad1d((3, 4))(conv1_1)
        conv1_1 = self.batchNormconv1(conv1_1)
        conv1_2 = F.relu(self.conv1_2(conv1_1))

        conv2_1 = nn.MaxPool1d(kernel_size=2, stride=2)(conv1_2)
        conv2_1 = self.drop1(conv2_1)
        conv2_1 = nn.ReflectionPad1d((3, 4))(conv2_1)
        conv2_2 = F.relu(self.conv2_1(conv2_1))
        conv2_2 = nn.ReflectionPad1d((3, 4))(conv2_2)
        conv2_2 = self.batchNormconv2(conv2_2)
        conv2_3 = F.relu(self.conv2_2(conv2_2))

        conv3_1 = nn.MaxPool1d(kernel_size=2, stride=2)(conv2_3)
        conv3_1 = self.drop1(conv3_1)
        conv3_1 = nn.ReflectionPad1d((3, 4))(conv3_1)
        conv3_2 = F.relu(self.conv3_1(conv3_1))
        conv3_2 = nn.ReflectionPad1d((3, 4))(conv3_2)
        conv3_3 = F.relu(self.conv3_2(conv3_2))
        conv3_3 = nn.ReflectionPad1d((3, 4))(conv3_3)
        conv3_3 = self.batchNormconv3(conv3_3)
        conv3_4 = F.relu(self.conv3_2(conv3_3))

        conv2_4_1 = self.convTranspose1(conv3_4)
        conv2_4_1 = self.drop1(conv2_4_1)
        conv2_4 = torch.cat((conv2_4_1, conv2_3), 1)
        conv2_4 = nn.ReflectionPad1d((3, 4))(conv2_4)
        conv2_5 = F.relu(self.conv2_5(conv2_4))
        conv2_5 = nn.ReflectionPad1d((3, 4))(conv2_5)
        conv2_5 = self.batchNormconv4(conv2_5)
        conv2_6 = F.relu(self.conv2_6(conv2_5))

        conv1_3_1 = self.convTranspose2(conv2_6)
        conv1_3_1 = self.drop1(conv1_3_1)
        conv1_3 = torch.cat((conv1_2, conv1_3_1), 1)
        conv1_3 = nn.ReflectionPad1d((3, 4))(conv1_3)
        conv1_3 = self.batchNormconv4(conv1_3)
        conv1_4 = F.relu(self.conv1_4(conv1_3))
        conv1_4 = nn.ReflectionPad1d((3, 4))(conv1_4)
        conv1_4 = F.relu(self.conv1_4_2(conv1_4))
        conv1_5 = F.relu(self.conv1_5(conv1_4))
        conv1_5 = self.drop2(conv1_5)
        conv1_6 = self.conv1_6(conv1_5)
        conv1_6 = conv1_6.view(conv1_6.size(0), -1)

        if self.as_head:
            fc1 = self.fc1(conv1_6)
            fc2 = self.fc2(fc1)
            out = self.fc3(fc2)
        else:
            out = self.fc(conv1_6)

        if self.is_mtl and not self.as_head:
            aot_y = self.aot_h(out)
            scale_y = self.scale_h(out)
            permute_y = self.permute_h(out)
            time_w_h = self.time_w_h(out)
            return aot_y, scale_y, permute_y, time_w_h

        elif self.is_simclr:
            return out

        elif self.is_eva:
            out = self.classifier(out)
            return out


        return out


class ElderNet(nn.Module):
    def __init__(self, feature_extractor, head='fc', non_linearity=True,  linear_model_input_size=1024,
                 linear_model_output_size=50, is_mtl=False, is_simclr=False, is_eva=False, is_dense=False):
        super(ElderNet, self).__init__()
        # Load the pretrained layers without classifier
        self.feature_extractor = feature_extractor
        self.is_mtl = is_mtl # multy task learning
        self.is_simclr = is_simclr
        self.is_eva = is_eva #evaluating mode (fine-tuning)
        self.is_dense = is_dense #dense labeling
        # Freeze the pretrained layers
        if not self.is_eva:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        # Add the small model
        self.head = head

        # Option 1: FC layers
        if self.head == 'fc':
            self.fc = LinearLayers(linear_model_input_size, linear_model_output_size, non_linearity)
        # Option 2: adding the unet layers
        elif self.head == 'unet':
            self.unet = Unet(as_head=True,is_eva=self.is_eva, is_mtl=self.is_mtl, is_simclr=self.is_simclr)

        if self.is_mtl:
            self.aot_h = Classifier(
                input_size=linear_model_output_size, output_size=2
            )
            self.scale_h = Classifier(
                input_size=linear_model_output_size, output_size=2
            )
            self.permute_h = Classifier(
                input_size=linear_model_output_size, output_size=2
            )
            self.time_w_h = Classifier(
                input_size=linear_model_output_size, output_size=2
            )

        # Indicate if we are in evaluation mode (and then add a classification head)
        if self.is_eva:
            self.classifier = Classifier(linear_model_output_size, 2)

        if self.is_dense:
            self.classifier = Classifier(linear_model_output_size, 300)

    def forward(self, x):
        features = self.feature_extractor(x)
        if self.head == 'fc':
            representation = self.fc(features.view(x.shape[0], -1))
        elif self.head == 'unet':
            features = torch.transpose(features, 2, 1)
            # features =  features.view(x.shape[0], -1)
            representation = self.unet(features)

        if self.is_mtl:
            aot_y = self.aot_h(representation)
            scale_y = self.scale_h(representation)
            permute_y = self.permute_h(representation)
            time_w_h = self.time_w_h(representation)

            return aot_y, scale_y, permute_y, time_w_h

        elif self.is_eva or self.is_dense:
            logits = self.classifier(representation)
            return logits

        elif self.is_simclr:
            return representation


def weight_init(self, mode="fan_out", nonlinearity="relu"):

    for m in self.modules():

        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(
                m.weight, mode=mode, nonlinearity=nonlinearity
            )

        elif isinstance(m, (nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)























