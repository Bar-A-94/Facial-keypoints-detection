from torchvision import models
import torch.nn as nn

resnet50 = models.resnet50(pretrained=True)
resnet50.fc = nn.Linear(2048, 30)
resnet50.__name__ = "resnet50"

efficientb4 = models.efficientnet_b4(pretrained=True)
efficientb4.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(2048, 30))
efficientb4.__name__ = "efficientb4"

inception_v3 = models.inception_v3(pretrained=True, aux_logits=False)
for param in inception_v3.parameters():
    param.requires_grad = False
inception_v3.fc = nn.Linear(2048, 30)
inception_v3.__name__ = "inception_v3"
