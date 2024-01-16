import torch
from tqdm import tqdm
import numpy as np
import csv
from utils import get_device, create_transforms, initialize_models
import argparse
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def load_data(path, transform, batch_size):
    dataset = ImageFolder(root=path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print("Loaded dataset:")
    print(f" - Number of images: {len(dataset)}")
    print(f" - Number of classes: {len(dataset.classes)}")
    print(f" - Class names: {dataset.classes}")
    print(f" - Batch size: {batch_size}")

    return loader

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def generate_class_predictions(model, dataloader, device, BATCH_SIZE):
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

        outputs = model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        class_predictions.append(predicted.cpu())

    return torch.cat(class_predictions), torch.cat(all_true_labels), image_names

def calculate_accuracy(predictions, true_labels):
    correct = (predictions == true_labels).sum().item()
    total = true_labels.size(0)
    accuracy = correct / total
    return accuracy

def main(train_data_path, run_path, batch_size, _type):
    device = get_device()
    print(f"Using device: {device}")

    _, model = initialize_models(device)
    transform = create_transforms()
    train_loader = load_data(train_data_path, transform, batch_size)

    model_path = f'{run_path}Inception3_epoch_48.pth'
    model = load_model(model, model_path)
    class_predictions, true_labels, image_names = generate_class_predictions(model, train_loader, device, batch_size)
    if _type == 'val':
        accuracy = calculate_accuracy(class_predictions, true_labels)
        print(f"Accuracy of the model: {accuracy * 100:.2f}%")

    class_names = ['nevus','others']
    with open(f"prediction_{_type}.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["image_name","Image Number", "class_number", "class_name"])
        for idx, pred in enumerate(class_predictions):
            image_path = image_names[idx]
            image_name = image_path.split('/')[-1]
            class_number = pred.item()
            class_name = class_names[class_number]
            csvwriter.writerow([image_name,idx, class_number, class_name])

    with open(f"detailed_prediction_{_type}.txt", "w") as f:
        for idx, (pred, true_label) in enumerate(zip(class_predictions, true_labels)):
            f.write(f"Image {idx}: Predicted Class {pred.item()}, True Class {true_label.item()}\n")

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Run the model prediction")

    # Add arguments
    parser.add_argument('--train_data_path', type=str, default='train_data', help='Path to the training data')
    parser.add_argument('--run_path', type=str, default='out/run_1/', help='Path to save or load the model')
    parser.add_argument('--type', type=str, choices=['val', 'test'], default='val', help='Type of dataset to use (validation or test)')
    parser.add_argument('--BATCH_SIZE', type=int, default=16, help='Batch size for model predictions')
