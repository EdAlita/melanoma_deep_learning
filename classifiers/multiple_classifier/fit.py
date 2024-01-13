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
from tqdm import tqdm
from torchsummary import summary
from collections import Counter
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import cohen_kappa_score



def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calculate_class_weights(dataset):
    class_counts = [0] * len(dataset.classes)

    for _, index in tqdm(dataset, desc="Calculating class weights"):
        class_counts[index] += 1
    
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    return class_weights / class_weights.sum()

def create_transforms():
    return transforms.Compose([
        transforms.Resize(299),  # Resize the images to 299 x 299 pixels
        transforms.CenterCrop(299),  # Crop the images to 299 x 299 pixels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def load_data(path, transform=None, class_weights=None,batch_size=BATCH_SIZE):
    if not os.path.exists(path):
        raise ValueError(f"The provided path {path} does not exist.")

    dataset = ImageFolder(root=path, transform=transform)

    # Create weights for each sample
    weights = class_weights[dataset.targets]
    sampler = WeightedRandomSampler(weights, len(dataset))

    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    image_count = Counter([dataset.targets[i] for i in range(len(dataset))])
    class_image_count = {dataset.classes[k]: v for k, v in image_count.items()}

    print("Loaded dataset:")
    print(f" - Number of images: {len(dataset)}")
    print(f" - Number of classes: {len(dataset.classes)}")
    print(f" - Class names: {dataset.classes}")
    for class_name, count in class_image_count.items():
        print(f" -- {class_name}: {count} images")
    print(f" - Batch size: {batch_size}")

    return loader

def initialize_models(device):
    inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    efficientnet_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

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

def train_model(model, optimizer, scheduler, train_loader, criterion, num_epochs, device, save_path):
    model_name = type(model).__name__
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        start_time = time.time()  # Start time for the epoch

        print(f"Training {model_name} - Epoch {epoch+1}/{num_epochs}")
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")

        model.train()
        running_loss = 0.0
        running_corrects = 0

        all_preds = []
        all_labels = []

        # Initialize tqdm progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch")

        for inputs, labels in progress_bar:
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                if model_name == 'Inception3':
                    outputs, aux_outputs = model(inputs)
                    loss1 = criterion(outputs, labels)
                    loss2 = criterion(aux_outputs, labels)
                    loss = loss1 + 0.4 * loss2
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                loss.backward()
                optimizer.step()
                scheduler.step()
                    
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Update the progress bar description with current loss and accuracy
            progress_bar.set_description(f"{model_name} Epoch {epoch+1} - Loss: {running_loss / len(train_loader.dataset):.4f}, Acc: {running_corrects.double() / len(train_loader.dataset):.4f}, Kappa: {cohen_kappa_score(all_labels, all_preds):.4f}")

        val_loss = running_loss / len(train_loader.dataset)
        
        if val_loss < best_loss:
            best_loss = running_loss
            torch.save(model.state_dict(), os.path.join(save_path, f'{model_name}_best.pth'))
            print(f'Saved improved model at epoch {epoch+1}')

        epoch_time = time.time() - start_time  # Calculate time taken for the epoch
        print(f'{model_name} - Epoch {epoch+1} Completed - Time: {epoch_time:.2f}s')

def main(number_epochs=10, save_dir='out/run_10/', train_dir='../../data_masks/train/', batch_size = 16):
    device = get_device()
    print(f"Using device: {device}")
    print(f"Train data: {train_dir} Out path: {save_dir}")

    transform = create_transforms()

    efficientnet_model, inception = initialize_models(device)

    #summary(resnet50_model)


    class_weights = calculate_class_weights(ImageFolder(root=train_data_path))
    train_loader = load_data(train_data_path, transform, class_weights)

    print(f'Class weigths: {class_weights}')
    criterion = nn.CrossEntropyLoss().to(device)

    optimizers = create_optimizers([inception, efficientnet_model])
    steps_per_epoch = len(train_loader)
    schedulers = create_schedulers(optimizers, steps_per_epoch)
    
    num_epochs = number_epochs
    
    model_save_folder = save_dir  # Replace with your folder path

    
    for model, optimizer, scheduler in zip([inception,efficientnet_model], optimizers, schedulers):
        train_model(model, optimizer, scheduler, train_loader, criterion, num_epochs, device,model_save_folder)


if __name__ == "__main__":
    main()
