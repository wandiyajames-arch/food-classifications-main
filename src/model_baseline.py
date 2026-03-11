import torch.nn as nn
import torchvision.models as models


def simple_cnn(num_classes):

    model = nn.Sequential(

        nn.Conv2d(3,32,3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32,64,3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),

        nn.Linear(64,128),
        nn.ReLU(),
        nn.Linear(128,num_classes)
    )

    return model


def resnet18_transfer(num_classes):
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model