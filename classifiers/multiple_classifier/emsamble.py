import torch
import torch.nn as nn
from tqdm import tqdm
import csv
from sklearn.metrics import cohen_kappa_score
from utils import get_device, create_transforms, initialize_models, load_data, load_models 
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

def generate_class_predictions(models, dataloader, device,batch_size):
    class_predictions = []
    all_true_labels = []
    image_names = []

    for batch_idx, (X_batch, y_batch) in tqdm(enumerate(dataloader), desc="Generating Class Predictions"):
        start_idx = batch_idx * batch_size
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

def main(train_data_path='../data_mult/train',run_path='out/run_1/',type='val',batch_size=16):
    device = get_device()
    print(f"Using device: {device}")

    inception , _ = initialize_models(device)


    transform = create_transforms()

    train_loader = load_data(train_data_path, transform,batch_size)
    #validation_loader = load_data(val_data_path, transform)

    model_paths = [
    f'{run_path}EfficientNet_best.pth', # Rank 1, Accuracy: 0.9228, Kappa: 0.8589
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
    class_predictions, true_labels, image_names = generate_class_predictions(models, train_loader, device,batch_size)
    final_predictions, vote_counts = maximum_voting(class_predictions)
    if type == 'val':
        accuracy = calculate_accuracy(final_predictions, true_labels)
        print(f"Accuracy of the ensemble: {accuracy * 100:.2f}%")
        kappa = cohen_kappa_score(final_predictions.numpy(), true_labels.numpy())
        print(f"Cohen's Kappa: {kappa:.4f}")
    
    class_names = ['bcc','mel','scc']
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
    parser = argparse.ArgumentParser(description="Run the ensemble model prediction")

    # Add arguments
    parser.add_argument('--train_data_path', type=str, default='train_data', help='Path to the training data')
    parser.add_argument('--run_path', type=str, default='out/run_1/', help='Path to save or load the models')
    parser.add_argument('--type', type=str, choices=['val', 'test'], default='val', help='Type of dataset to use (validation or test)')
    parser.add_argument('--BATCH_SIZE', type=int, default=16, help='Batch size for model predictions')

    # Parse the arguments
    args = parser.parse_args()

    # Call main function with parsed arguments
    main(args.train_data_path, args.run_path, args.type, args.BATCH_SIZE)
