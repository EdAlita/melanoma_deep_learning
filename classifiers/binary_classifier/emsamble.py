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


BATCH_SIZE = 16
train_data_path = '../../data/val/'


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_transforms():
    return transforms.Compose([
        transforms.Resize((299, 299)),  # Adjusted for Inception V3
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def validate_model(model, validation_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_samples = 0

    with torch.no_grad():  # No need to track gradients
        for X_batch, y_batch in tqdm(validation_loader, desc="Validating"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)

    average_loss = total_loss / total_samples
    return average_loss

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

# Define the meta-learner model
class MetaLearnerModel(nn.Module):
    def __init__(self, num_base_models):
        super(MetaLearnerModel, self).__init__()
        self.fc = nn.Linear(num_base_models, 1)  # Adjust the output features as needed

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

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
    first_batch = True
    all_true_labels = []

    for X_batch, y_batch in tqdm(dataloader, desc="Generating Class Predictions"):
        if first_batch:
            all_true_labels = y_batch.numpy()  # Store true labels from the first batch
            first_batch = False
        else:
            all_true_labels = np.concatenate((all_true_labels, y_batch.numpy()), axis=0)  # Concatenate subsequent true labels

        X_batch = X_batch.to(device)
        batch_predictions = []
        for model in models:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)  # Get the class with the highest score
            batch_predictions.append(predicted.cpu())
        class_predictions.append(torch.stack(batch_predictions, dim=1))

    return torch.cat(class_predictions, dim=0), torch.tensor(all_true_labels)

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

def main():
    device = get_device()
    print(f"Using device: {device}")

    inception, resnet50_model = initialize_models(device)


    transform = create_transforms()

    train_loader = load_data(train_data_path, transform)
    #validation_loader = load_data(val_data_path, transform)


    model_paths = [
    "out/run_1/Inception3_epoch_48.pth",
    "out/run_1/Inception3_epoch_43.pth",
    "out/run_1/Inception3_epoch_42.pth",
    "out/run_1/Inception3_epoch_44.pth",
    "out/run_1/Inception3_epoch_59.pth",
    "out/run_1/Inception3_epoch_56.pth",
    "out/run_1/Inception3_epoch_36.pth",
    "out/run_1/Inception3_epoch_47.pth",
    "out/run_1/Inception3_epoch_58.pth",
    "out/run_1/Inception3_epoch_60.pth",
    "out/run_1/Inception3_epoch_35.pth",
    "out/run_1/Inception3_epoch_37.pth"]

    models = load_models(inception, model_paths)
    class_predictions, true_labels = generate_class_predictions(models, train_loader, device)
    final_predictions, vote_counts = maximum_voting(class_predictions)

    # Calculate accuracy
    accuracy = calculate_accuracy(final_predictions, true_labels)
    print(f"Accuracy of the ensemble: {accuracy * 100:.2f}%")

    # Save predictions to a file
    with open("predictions.txt", "w") as f:
        for idx, (pred, true_label) in enumerate(zip(final_predictions, true_labels)):
            f.write(f"Image {idx}: Predicted Class {pred.item()}, True Class {true_label.item()}, Vote counts: {vote_counts[idx].tolist()}\n")

if __name__ == "__main__":
    main()
