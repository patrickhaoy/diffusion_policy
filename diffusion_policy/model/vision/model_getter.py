import torch
import torchvision

def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model

def get_dinov3(name='vit_small_patch16_dinov3', pretrained=True, **kwargs):
    """
    name: vit_small_patch16_dinov3, vit_small_plus_patch16_dinov3,
          vit_base_patch16_dinov3, vit_large_patch16_dinov3, etc.
    Returns a ViT that outputs (B, embed_dim) feature vectors.
    """
    import timm
    model = timm.create_model(name, pretrained=pretrained, num_classes=0, **kwargs)
    return model
