import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import inception_v3, resnet50, Inception_V3_Weights, ResNet50_Weights
from cnn import CustomClassifier
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np
import csv


BATCH_SIZE = 16
train_data = '../../data/val/'


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

def generate_class_predictions(models, dataloader, device):
    class_predictions = []
    all_true_labels = []
    image_names = []

    for X_batch, y_batch in tqdm(dataloader, desc="Generating Class Predictions"):
        y_batch_numpy = y_batch.cpu().numpy()
        image_names.extend([dataloader.dataset.samples[i][0] for i in y_batch_numpy])
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        all_true_labels.append(y_batch.cpu())
        batch_predictions = []
        for model in models:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            batch_predictions.append(predicted.cpu())
        class_predictions.append(torch.stack(batch_predictions, dim=1))

    return torch.cat(class_predictions, dim=0), torch.cat(all_true_labels, dim=0), image_names

def maximum_voting(predictions):
    vote_results = []
    vote_counts = []  # To store the vote counts for each class
    for i in range(predictions.shape[0]):
        votes = predictions[i]
        vote_count = torch.bincount(votes, minlength=2)  # NUM_CLASSES should be replaced with actual number
        most_voted_class = torch.argmax(vote_count)
        vote_results.append(most_voted_class)
        vote_counts.append(vote_count)
    return torch.stack(vote_results), torch.stack(vote_counts)

def calculate_accuracy(predictions, true_labels):
    correct = (predictions == true_labels).sum().item()
    total = true_labels.size(0)
    accuracy = correct / total
    return accuracy

def main(train_data_path=train_data,run_path='out/run_1/'):
    device = get_device()
    print(f"Using device: {device}")

    inception, resnet50_model = initialize_models(device)


    transform = create_transforms()

    train_loader = load_data(train_data_path, transform)
    #validation_loader = load_data(val_data_path, transform)


    model_paths = [
    f'{run_path}Inception3_epoch_48.pth',
    f'{run_path}Inception3_epoch_43.pth',
    f'{run_path}Inception3_epoch_42.pth',
    f'{run_path}Inception3_epoch_44.pth',
    f'{run_path}Inception3_epoch_59.pth',
    f'{run_path}Inception3_epoch_56.pth',
    f'{run_path}Inception3_epoch_36.pth',
    f'{run_path}Inception3_epoch_47.pth',
    f'{run_path}Inception3_epoch_58.pth',
    f'{run_path}Inception3_epoch_60.pth',
    f'{run_path}Inception3_epoch_35.pth',
    f'{run_path}Inception3_epoch_37.pth']

    models = load_models(inception, model_paths)
    class_predictions, true_labels, image_names = generate_class_predictions(models, train_loader, device)
    final_predictions, vote_counts = maximum_voting(class_predictions)

    accuracy = calculate_accuracy(final_predictions, true_labels)
    print(f"Accuracy of the ensemble: {accuracy * 100:.2f}%")

    class_names = train_loader.dataset.classes
    with open("predictions.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["image_name", "class_number", "class_name"])
        for idx, pred in enumerate(final_predictions):
            image_path = image_names[idx]
            image_name = image_path.split('/')[-1]
            class_number = pred.item()
            class_name = class_names[class_number]
            csvwriter.writerow([image_name, class_number, class_name])

    with open("detailed_predictions.txt", "w") as f:
        for idx, (pred, true_label) in enumerate(zip(final_predictions, true_labels)):
            f.write(f"Image {idx}: Predicted Class {pred.item()}, True Class {true_label.item()}, Vote counts: {vote_counts[idx].tolist()}\n")

if __name__ == "__main__":
    main()
