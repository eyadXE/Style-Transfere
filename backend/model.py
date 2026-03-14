import torch
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).to(device).eval()


def get_feature_maps(img, net):

    featuremaps = []
    featurenames = []
    conv_idx = 0

    for layer in net.features:
        img = layer(img)

        if isinstance(layer, torch.nn.Conv2d):
            featuremaps.append(img)
            featurenames.append(f"ConvLayer_{conv_idx}")
            conv_idx += 1

    return featuremaps, featurenames


def gram_matrix(F):

    _, c, h, w = F.shape
    F = F.view(c, h * w)

    G = torch.mm(F, F.t())
    return G / (c * h * w)