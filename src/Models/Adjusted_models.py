from torchvision import models
import torch.nn as nn

resnet50 = models.resnet50(pretrained=True)
resnet50.fc = nn.Linear(2048, 30)
resnet50.__name__ = "resnet50"

mobilenet_v3 = models.mobilenet_v3_large(pretrained=True)
mobilenet_v3.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, 30))
mobilenet_v3.__name__ = "mobilenet_v3"

efficientb3 = models.efficientnet_b3(pretrained=True)
efficientb3.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(1536, 30))
efficientb3.__name__ = "efficientb3"

inception_v3 = models.inception_v3(pretrained=True, aux_logits=False)
for param in inception_v3.parameters():
    param.requires_grad = False
inception_v3.fc = nn.Linear(2048, 30)
inception_v3.__name__ = "inception_v3"
