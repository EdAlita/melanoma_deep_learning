import torch
from torchvision import transforms
from torchvision.models import inception_v3, efficientnet_b0, Inception_V3_Weights, EfficientNet_B0_Weights
from cnn import CustomClassifier
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_transforms():
    return transforms.Compose([
        transforms.Resize((299, 299)),  # Adjusted for Inception V3
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def initialize_models(device):
    try: 
        inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        efficientnet_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    except RuntimeError as e:
        inception_model = inception_v3(pretrained=True)
        efficientnet_model = efficientnet_b0(pretrained=True)
        
    inception_model.fc = CustomClassifier(inception_model.fc.in_features)
    efficientnet_model.classifier[1] = CustomClassifier(efficientnet_model.classifier[1].in_features)

    inception_model = inception_model.to(device)
    efficientnet_model = efficientnet_model.to(device)
    
    print("Initialized models:")
    print(" - Inception V3 with custom classifier")
    print(" - EfficientNet B0 with custom classifier")

    return efficientnet_model, inception_model

def load_data(path, transform):
    dataset = ImageFolder(root=path, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Loaded dataset:")
    print(f" - Number of images: {len(dataset)}")
    print(f" - Number of classes: {len(dataset.classes)}")
    print(f" - Class names: {dataset.classes}")
    print(f" - Batch size: {BATCH_SIZE}")

    return loader

def load_models(model, model_paths):
    models = []
    for path in model_paths:
        model = model
        model.load_state_dict(torch.load(path))
        model.eval()
        models.append(model)
    return models