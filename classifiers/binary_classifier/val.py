import torch
from tqdm.auto import tqdm
import os
import argparse
from utils import get_device, create_transforms, initialize_models, load_data


def load_checkpoint(filepath, device, model):
    checkpoint = torch.load(filepath, map_location=torch.device(device))
    model = model  # Initialize your model
    model.load_state_dict(checkpoint)  # Load saved state_dict
    return model

def evaluate_model(model, val_loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)  # Move to the same device as the model
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def initialize_models(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    print(checkpoint.keys())

def evaluating_best_classifier(val_dir, root_out, batch_size=16):
    device = get_device()
    print(f"Using device: {device}")
    
    transform = create_transforms()
    val_loader = load_data(val_dir, transform)

    model_paths = {
        'EfficientNet': os.path.join(root_out, 'EfficientNet_best.pth'),
        'Inception3': os.path.join(root_out, 'Inception3_best.pth')
    }

    efficientnet,inception = initialize_models(device)

    for model_name, path in model_paths.items():
        if model_name == 'Inception3':
            model = load_checkpoint(path, device, inception)
        elif model_name == 'EfficientNet':
            model = load_checkpoint(path, device, efficientnet)

        accuracy = evaluate_model(model, val_loader, device)
        print(f"Model: {model_name}, Path: {path}, Accuracy: {accuracy}")

def main(num_classifiers=60,val_dir = '../../data/val/',root_out = 'out/run_4/',batch_size=16):
    
    device = get_device()
    print(f"Using device: {device}")
    
    transform = create_transforms()
    val_loader = load_data(val_dir, transform,batch_size)

    model_paths = [os.path.join(root_out,f'Inception3_epoch_{i}.pth') for i in range(num_classifiers)]

    efficientnet_model, inception_model = initialize_models(device)

    model_accuracies = {}
    for path in tqdm(model_paths, desc="Model Evaluation Progress"):
        model = load_checkpoint(path, device, inception)
        accuracy = evaluate_model(model, val_loader, device)
        model_accuracies[path] = accuracy
        
    sorted_models = sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True)

    top_models = sorted_models[:5]
    print("Top 12 Models:")
    for i, model in enumerate(top_models, start=1):
        print(f"Rank {i}: Model Path: {model[0]}, Accuracy: {model[1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nomal or best models validation.')
    parser.add_argument('--mode', choices=['train', 'best'], required=True, help='Choose "normal" to run main or "best" to run evaluating_best_classifier [Normal options use if you have results from v 1.1]')
    
    # Arguments for both functions
    parser.add_argument('--val_dir', type=str, default='../../data/val/', help='Path to validation data directory')
    parser.add_argument('--root_out', type=str, default='out/run_4/', help='Root directory for model checkpoints')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for data loading')

    # Arguments specific to main()
    parser.add_argument('--num_classifiers', type=int, default=60, help='Number of classifiers for main function [Used in normal DEPRECATED]')

    args = parser.parse_args()

    if args.mode == 'normal':
        main(args.num_classifiers, args.val_dir, args.root_out, args.batch_size)
    elif args.mode == 'best':
        evaluating_best_classifier(args.val_dir, args.root_out, args.batch_size)
