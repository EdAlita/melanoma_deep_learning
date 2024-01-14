import torch
from tqdm import tqdm
import numpy as np
import csv
from utils import get_device, create_transforms, initialize_models, load_data
import argparse

train_data = '../../data/val/'

def load_models(model, model_paths):
    models = []
    for path in model_paths:
        model = model
        model.load_state_dict(torch.load(path))
        model.eval()
        models.append(model)
    return models

def generate_class_predictions(models, dataloader, device,BATCH_SIZE ):
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

def main(train_data_path, run_path, model_type, batch_size):
    device = get_device()
    print(f"Using device: {device}")

    _ , inception= initialize_models(device)


    transform = create_transforms()

    train_loader = load_data(train_data_path, transform)
    #validation_loader = load_data(val_data_path, transform)


    model_paths = [
        f'{run_path}Inception3_epoch_8.pth',
        #f'{run_path}Inception3_epoch_21.pth',
        #f'{run_path}Inception3_epoch_7.pth',
        #f'{run_path}Inception3_epoch_25.pth',
        #f'{run_path}Inception3_epoch_13.pth',
        #f'{run_path}Inception3_epoch_31.pth',
        ]

    models = load_models(inception, model_paths)
    class_predictions, true_labels, image_names = generate_class_predictions(models, train_loader, device, BATCH_SIZE)
    final_predictions, vote_counts = maximum_voting(class_predictions)
    if type == 'val':
        accuracy = calculate_accuracy(final_predictions, true_labels)
        print(f"Accuracy of the ensemble: {accuracy * 100:.2f}%")

    class_names = train_loader.dataset.classes
    with open(f"predictions_{type}.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["image_name", "class_number", "class_name"])
        for idx, pred in enumerate(final_predictions):
            image_path = image_names[idx]
            image_name = image_path.split('/')[-1]
            class_number = pred.item()
            class_name = class_names[class_number]
            csvwriter.writerow([image_name, class_number, class_name])

    with open(f"detailed_predictions_{type}.txt", "w") as f:
        for idx, (pred, true_label) in enumerate(zip(final_predictions, true_labels)):
            f.write(f"Image {idx}: Predicted Class {pred.item()}, True Class {true_label.item()}, Vote counts: {vote_counts[idx].tolist()}\n")

if __name__ == "__main__":
    # Create the parser
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
