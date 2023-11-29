import torch
import torch.nn.functional as F
import torchvision.models as tvm


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, feature_layers, weights, distance=F.l1_loss, resize=True, normalized=True):
        super(VGGPerceptualLoss, self).__init__()

        assert len(feature_layers) == len(weights)

        self.feature_layers_weights = dict(zip(feature_layers, weights))

        self.features = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1).features[:max(feature_layers)]
        self.features.requires_grad_(False)
        self.features.eval()

        self.distance = distance

        self.resize = resize
        self.normalized = normalized
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        if self.normalized:
            input = (input-self.mean) / self.std
            target = (target-self.mean) / self.std

        if self.resize:
            input = F.interpolate(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = F.interpolate(target, mode='bilinear', size=(224, 224), align_corners=False)

        loss = 0
        x, y = input, target
        for i, block in enumerate(self.features):
            x, y = block(x), block(y)
            if i in self.feature_layers_weights:
                loss += self.distance(x, y) * self.feature_layers_weights[i]
        return loss
