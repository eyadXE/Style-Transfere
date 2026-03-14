# backend/style_transfer.py
import torch
import torch.nn as nn
from torchvision import models, transforms as T
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pretrained AlexNet
alexnet = models.alexnet(pretrained=True).to(device).eval()

# Image transform
transform = T.Compose([
    T.Resize((256, 256)),  # small size for web
    T.ToTensor(),
])

def get_feature_maps(img, net):
    featuremaps = []
    featurenames = []
    conv_idx = 0
    x = img
    for layernum in range(len(net.features)):
        x = net.features[layernum](x)
        if isinstance(net.features[layernum], nn.Conv2d):
            featuremaps.append(x)
            featurenames.append(f"ConvLayer_{conv_idx}")
            conv_idx += 1
    return featuremaps, featurenames

def gram_matrix(F):
    _, C, H, W = F.shape
    F = F.view(C, H*W)
    return torch.mm(F, F.t()) / (C*H*W)

def run_style_transfer(content_img: Image.Image, style_img: Image.Image,
                       numepochs=500, styleScaling=5e4, lr=0.001):
    
    # Transform images
    img_content = transform(content_img).unsqueeze(0).to(device)
    img_style   = transform(style_img).unsqueeze(0).to(device)

    # Initialize target
    target = img_content.clone().requires_grad_(True).to(device)

# Detach features to prevent backprop through them
    contentFeatureMaps, _ = get_feature_maps(img_content, alexnet)
    contentFeatureMaps = [f.detach() for f in contentFeatureMaps]
    
    styleFeatureMaps, _   = get_feature_maps(img_style, alexnet)
    styleFeatureMaps = [f.detach() for f in styleFeatureMaps]

    layers4content = ['ConvLayer_0']
    layers4style   = ['ConvLayer_0','ConvLayer_1','ConvLayer_2','ConvLayer_3','ConvLayer_4']
    weights4style  = [1, 0.8, 0.6, 0.4, 0.2]

    optimizer = torch.optim.RMSprop([target], lr=lr)

    for epoch in range(numepochs):
        # Compute target features fresh every iteration
        targetFeatureMaps, targetFeatureNames = get_feature_maps(target, alexnet)

        contentLoss = 0
        styleLoss = 0

        for i, name in enumerate(targetFeatureNames):
            if name in layers4content:
                contentLoss += torch.mean((targetFeatureMaps[i] - contentFeatureMaps[i])**2)
            if name in layers4style:
                Gtarget = gram_matrix(targetFeatureMaps[i])
                Gstyle  = gram_matrix(styleFeatureMaps[i])
                styleLoss += torch.mean((Gtarget - Gstyle)**2) * weights4style[layers4style.index(name)]

        total_loss = styleScaling*styleLoss + contentLoss

        optimizer.zero_grad()
        total_loss.backward(retain_graph=False)  # <--- only one backward per iteration
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{numepochs}] Loss: {total_loss.item():.2f}")

    return target.detach().squeeze(0)  # C,H,W