import torch.nn as nn
from torchvision import models



def create_model(num_classes, model_name='mobilenet_v2', fine_tune=False):
    """
    Download pretrained model and update last layer or fine-tuning
    Args:
    num_classes (int): Class number on dataset
    model_name (str): name of the selected model
    fine_tune (bool): Determines fine-tuning or last layer updating

    Returns:
    torch.nn.Module: Customized model object
    """
    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # If fine_tune is false, freeze all layers except the last one
        for param in model.parameters():
            param.requires_grad = fine_tune
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
        # The last layer should always be trainable
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif model_name == 'mobilenet_v3':
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = fine_tune
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = fine_tune
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        print(f"Error: '{model_name}' does not support. Please select 'mobilenet_v2', 'mobilenet_v3' or 'resnet18'.")

    return model

