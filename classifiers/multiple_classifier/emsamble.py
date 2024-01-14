import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import inception_v3, efficientnet_b0, Inception_V3_Weights, EfficientNet_B0_Weights
from cnn import CustomClassifier
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np
import csv
from sklearn.metrics import cohen_kappa_score

BATCH_SIZE = 16
train_data = '../../data_mult/val/'


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

def generate_class_predictions(models, dataloader, device):
    class_predictions = []
    all_true_labels = []
    image_names = []

    for batch_idx, (X_batch, y_batch) in tqdm(enumerate(dataloader), desc="Generating Class Predictions"):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = start_idx + X_batch.size(0)
        batch_image_names = [dataloader.dataset.samples[i][0] for i in range(start_idx, end_idx)]
        image_names.extend(batch_image_names)

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
        vote_count = torch.bincount(votes, minlength=3)  # NUM_CLASSES should be replaced with actual number
        most_voted_class = torch.argmax(vote_count)
        vote_results.append(most_voted_class)
        vote_counts.append(vote_count)
    return torch.stack(vote_results), torch.stack(vote_counts)

def calculate_accuracy(predictions, true_labels):
    correct = (predictions == true_labels).sum().item()
    total = true_labels.size(0)
    accuracy = correct / total
    return accuracy

def main(train_data_path=train_data,run_path='out/run_1/',type='val'):
    device = get_device()
    print(f"Using device: {device}")

    _ , inception= initialize_models(device)


    transform = create_transforms()

    train_loader = load_data(train_data_path, transform)
    #validation_loader = load_data(val_data_path, transform)

    model_paths = [
    f'{run_path}Inception3_epoch_31.pth', # Rank 1, Accuracy: 0.9228, Kappa: 0.8589
    #f'{run_path}Inception3_epoch_32.pth', # Rank 2, Accuracy: 0.9181, Kappa: 0.8497
    #f'{run_path}Inception3_epoch_28.pth', # Rank 3, Accuracy: 0.9173, Kappa: 0.8474
    #f'{run_path}Inception3_epoch_22.pth', # Rank 4, Accuracy: 0.9165, Kappa: 0.8474
    #f'{run_path}Inception3_epoch_36.pth', # Rank 5, Accuracy: 0.9165, Kappa: 0.8471
    #f'{run_path}Inception3_epoch_30.pth', # Rank 6, Accuracy: 0.9165, Kappa: 0.8468
    #f'{run_path}Inception3_epoch_20.pth', # Rank 7, Accuracy: 0.9165, Kappa: 0.8466
    #f'{run_path}Inception3_epoch_38.pth', # Rank 8, Accuracy: 0.9150, Kappa: 0.8439
    #f'{run_path}Inception3_epoch_21.pth', # Rank 9, Accuracy: 0.9150, Kappa: 0.8435
    #f'{run_path}Inception3_epoch_26.pth', # Rank 10, Accuracy: 0.9142, Kappa: 0.8433
    #f'{run_path}Inception3_epoch_27.pth', # Rank 11, Accuracy: 0.9142, Kappa: 0.8431
    #f'{run_path}Inception3_epoch_16.pth', # Rank 12, Accuracy: 0.9142, Kappa: 0.8429
    ]


    models = load_models(inception, model_paths)
    class_predictions, true_labels, image_names = generate_class_predictions(models, train_loader, device)
    final_predictions, vote_counts = maximum_voting(class_predictions)
    if type == 'val':
        accuracy = calculate_accuracy(final_predictions, true_labels)
        print(f"Accuracy of the ensemble: {accuracy * 100:.2f}%")
        kappa = cohen_kappa_score(final_predictions.numpy(), true_labels.numpy())
        print(f"Cohen's Kappa: {kappa:.4f}")
    
    class_names = train_loader.dataset.classes
    with open(f"predictions_mult_{type}.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["image_name", "class_number", "class_name"])
        for idx, pred in enumerate(final_predictions):
            image_path = image_names[idx]
            image_name = image_path.split('/')[-1]
            class_number = pred.item()
            class_name = class_names[class_number]
            csvwriter.writerow([image_name, class_number, class_name])

    with open(f"detailed_predictions_mult_{type}.txt", "w") as f:
        for idx, (pred, true_label) in enumerate(zip(final_predictions, true_labels)):
            f.write(f"Image {idx}: Predicted Class {pred.item()}, True Class {true_label.item()}, Vote counts: {vote_counts[idx].tolist()}\n")

if __name__ == "__main__":
    main()
