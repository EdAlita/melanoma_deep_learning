import torch
from torch.utils.data import DataLoader
from torchvision.models import inception_v3, resnet50, Inception_V3_Weights, ResNet50_Weights
from torchvision import transforms
from torchvision.datasets import ImageFolder
from cnn import CustomClassifier
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score


BATCH_SIZE = 32
validation_data_path = '../../data_mult/val/'

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_transforms():
    return transforms.Compose([
        transforms.Resize((299, 299)),  # Adjusted for Inception V3
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def initialize_models(device):
    inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    resnet50_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    inception_model.fc = CustomClassifier(inception_model.fc.in_features)
    resnet50_model.fc = CustomClassifier(resnet50_model.fc.in_features)

    inception_model = inception_model.to(device)
    resnet50_model = resnet50_model.to(device)
    
    print("Initialized models:")
    print(" - Inception V3 with custom classifier")
    print(" - ResNet50 with custom classifier")

    return inception_model, resnet50_model

def load_data(path, transform):
    dataset = ImageFolder(root=path, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    image_count = Counter([dataset.targets[i] for i in range(len(dataset))])
    class_image_count = {dataset.classes[k]: v for k, v in image_count.items()}

    print("Loaded dataset:")
    print(f" - Number of images: {len(dataset)}")
    print(f" - Number of classes: {len(dataset.classes)}")
    print(f" - Class names: {dataset.classes}")
    for class_name, count in class_image_count.items():
        print(f" -- {class_name}: {count} images")
    print(f" - Batch size: {BATCH_SIZE}")

    return loader

def load_checkpoint(filepath, device, model):
    checkpoint = torch.load(filepath, map_location=torch.device(device))
    model = model  # Initialize your model
    model.load_state_dict(checkpoint)  # Load saved state_dict
    return model


def evaluate_model(model, val_loader, device):
    model.eval()
    total = 0
    correct = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    accuracy = correct / total
    kappa = cohen_kappa_score(all_labels, all_predictions)
    return accuracy, kappa


def inspect_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    print(checkpoint.keys())


def main():
    device = get_device()
    print(f"Using device: {device}")
    
    #inspect_checkpoint("out/run_1/Inception3_epoch_0.pth")

    transform = create_transforms()

    # Load your validation dataset
    val_loader = load_data(validation_data_path, transform)

    # Path to your model checkpoints
    model_paths = [f"out/run_1/Inception3_epoch_{i}.pth" for i in range(60)]

    inception, resnet50_model = initialize_models(device)


    model_scores = {}
    for path in tqdm(model_paths, desc="Model Evaluation Progress"):
        model = load_checkpoint(path, device, inception)
        accuracy, kappa = evaluate_model(model, val_loader, device)
        model_scores[path] = (accuracy, kappa)

    

    #Sort models by accuracy
    sorted_models = sorted(model_scores.items(), key=lambda x: (x[1][1], x[1][0]), reverse=True)

    top_models = sorted_models[:12]
    print("Top Models Based on Kappa Score:")
    for i, (path, (accuracy, kappa)) in enumerate(top_models, start=1):
        print(f"Rank {i}: Model Path: {path}, Accuracy: {accuracy}, Kappa: {kappa}")

if __name__ == "__main__":
    main()
