import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
import os
import time
from tqdm.auto import tqdm
import argparse
from utils import get_device, create_transforms, initialize_models, load_data

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

def main(number_epochs=10, save_dir='out/run_10/', train_dir='../../data_masks/train/', batch_size = 16):
    device = get_device()
    print(f"Using device: {device}")
    print(f"Train data: {train_dir} Out path: {save_dir}")
    transform = create_transforms()
    train_loader = load_data(train_dir, transform, batch_size)

    efficientnet_model,inception = initialize_models(device)

    optimizers = create_optimizers([inception, efficientnet_model])
    steps_per_epoch = len(train_loader)
    schedulers = create_schedulers(optimizers, steps_per_epoch)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    num_epochs = number_epochs
    
    model_save_folder = save_dir

    for model, optimizer, scheduler in zip([inception, efficientnet_model], optimizers, schedulers):
        train_model(model, optimizer, scheduler, train_loader, criterion, num_epochs, device, model_save_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with command-line arguments.')
    parser.add_argument('--number_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--save_dir', type=str, default='out/run_10/', help='Directory to save the model')
    parser.add_argument('--train_dir', type=str, default='../../data_masks/train/', help='Path to training data directory')
    parser.add_argument('--batch_size', type=int, default=16,help='Number of batch size to load data')

    args = parser.parse_args()
    main(number_epochs=args.number_epochs, save_dir=args.save_dir, train_dir = args.train_dir, batch_size=args.batch_size)    
       
