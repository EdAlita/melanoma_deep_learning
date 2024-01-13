from torchvision.models import inception_v3, efficientnet_b0, Inception_V3_Weights, EfficientNet_B0_Weights

import torch
import torch.nn as nn
import torch.optim as optim
from cnn import CustomClassifier
from torch.optim.lr_scheduler import CyclicLR
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import time
from tqdm.auto import tqdm
from torchsummary import summary

BATCH_SIZE = 16
train_data_path = '../../data/train/'

# from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
# from torchvision.models._api import WeightsEnum
# from torch.hub import load_state_dict_from_url

# def get_state_dict(self, *args, **kwargs):
#     kwargs.pop("check_hash")
#     return load_state_dict_from_url(self.url, *args, **kwargs)
# WeightsEnum.get_state_dict = get_state_dict


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_transforms():
    return transforms.Compose([
        transforms.Resize(299),  # Resize the images to 299 x 299 pixels
        transforms.CenterCrop(299),  # Crop the images to 299 x 299 pixels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def load_data(path, transform):
    dataset = ImageFolder(root=path, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Loaded dataset:")
    print(f" - Number of images: {len(dataset)}")
    print(f" - Number of classes: {len(dataset.classes)}")
    print(f" - Class names: {dataset.classes}")
    print(f" - Batch size: {BATCH_SIZE}")

    return loader

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

def create_optimizers(models):
    return [optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0, amsgrad=False) for model in models]

def create_schedulers(optimizers, steps_per_epoch):
    base_lr, max_lr = 0.00001, 0.0001
    return [CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=4*steps_per_epoch, mode='triangular2', cycle_momentum=False) for optimizer in optimizers]

def train_model(model, optimizer, scheduler, train_loader, criterion, num_epochs, device, save_path, early_stopping_threshold=0.01, patience=5):
    model_name = type(model).__name__
    best_loss = float('inf')

    for epoch in range(num_epochs):
        start_time = time.time()  # Start time for the epoch

        print(f"Training {model_name} - Epoch {epoch+1}/{num_epochs}")
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")

        model.train()
        running_loss = 0.0
        running_corrects = 0

        # Initialize tqdm progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch")

        for inputs, labels in progress_bar:
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
            # Inception model outputs a tuple in training mode
                if type(model).__name__ == 'Inception3':
                    outputs, aux_outputs = model(inputs)
                else:
                    outputs = model(inputs)
                
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()
                scheduler.step()
                    
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            # Update the progress bar description with current loss and accuracy
            progress_bar.set_description(f"{model_name} Epoch {epoch+1} - Loss: {running_loss / len(train_loader.dataset):.4f}, Acc: {running_corrects.double() / len(train_loader.dataset):.4f}")

        val_loss = running_loss / len(train_loader.dataset)
        
        if val_loss < best_loss:
            best_loss = running_loss
            torch.save(model.state_dict(), os.path.join(save_path, f'{model_name}_best.pth'))
            print(f'Saved improved model at epoch {epoch+1}')

        epoch_time = time.time() - start_time  # Calculate time taken for the epoch
        print(f'{model_name} - Epoch {epoch+1} Completed - Time: {epoch_time:.2f}s')

def main(number_epochs=10, save_dir='out/run_9/', train_dir=train_data_path):

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Train data: {train_dir} Out path: {save_dir}")
    transform = create_transforms()
    train_loader = load_data(train_dir, transform)

    inception, efficientnet_model = initialize_models(device)

    optimizers = create_optimizers([inception, efficientnet_model])
    steps_per_epoch = len(train_loader)
    schedulers = create_schedulers(optimizers, steps_per_epoch)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    num_epochs = number_epochs
    
    model_save_folder = save_dir

    for model, optimizer, scheduler in zip([inception, efficientnet_model], optimizers, schedulers):
        train_model(model, optimizer, scheduler, train_loader, criterion, num_epochs, device, model_save_folder)

if __name__ == "__main__":
    main()    
       
