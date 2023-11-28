import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import inception_v3, resnet50, Inception_V3_Weights, ResNet50_Weights
from cnn import CustomClassifier
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


BATCH_SIZE = 32
train_data_path = '../../data/train/'


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

def generate_stacked_predictions(models, dataloader, device):
    stacked_predictions = []
    for X_batch, _ in tqdm(dataloader, desc="Generating Predictions"):
        X_batch = X_batch.to(device)
        batch_predictions = []
        for model in models:
            predictions = model(X_batch)
            batch_predictions.append(predictions.detach())
        stacked_batch = torch.stack(batch_predictions, dim=1)
        stacked_predictions.append(stacked_batch)
    
    return torch.cat(stacked_predictions, dim=0)


def train_meta_learner(dataloader, num_epochs=10, device=None):
    # Assuming the first batch to infer number of base models
    sample_inputs, _ = next(iter(dataloader))
    meta_learner = MetaLearnerModel(sample_inputs.size(1)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(meta_learner.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        meta_learner.train()  # Set the model to training mode
        for inputs, targets in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = meta_learner(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    return meta_learner



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

    models = load_models(inception,model_paths)
    stacked_predictions = generate_stacked_predictions(models, train_loader, device)
    
    y_train = []  # Load your y_train data appropriately
    meta_dataset = TensorDataset(stacked_predictions, torch.tensor(y_train, dtype=torch.float32))
    meta_loader = DataLoader(meta_dataset, batch_size=BATCH_SIZE, shuffle=True)

    meta_learner = train_meta_learner(meta_loader, device=device)


if __name__ == "__main__":
    main()
