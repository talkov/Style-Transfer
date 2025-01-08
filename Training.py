from tqdm import tqdm  # Import tqdm for progress tracking
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch
import wandb  # Uncomment wandb import
import os
from config import get_options
from model import StyleClassifier
from dataset import create_loaders, preprocess_function, make_dataset
import torchvision.transforms as transforms
import random


def train_model(model, train_loader, valid_loader, args):
    # Load model if specified
    if args.load_model:
        model = torch.load(args.load_model_path)
    model = model.to(args.device)

    best_model = copy.deepcopy(model)
    MIN_loss = np.inf
    max_val = 0
    best_epoch = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.scheduler_factor, patience=args.scheduler_patience, verbose=True)



    # Initialize Weights & Biases tracking
    if args.WANDB_TRACKING:
        wandb.login(key="")
        wandb.init(project="Style Classification Task")
        
    for epoch in range(1, args.num_epochs + 1):
        print(f"Epoch {epoch}/{args.num_epochs}\n")

        # Training Phase
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        with tqdm(train_loader, desc=f"Training Epoch {epoch}") as pbar:
            for batch in pbar:
                images, labels = batch["image"].to(args.device), batch["label"].to(args.device)

            
                # augmented_90 = torch.stack([transforms.functional.rotate(img, 90) for img in images])
                # augmented_180 = torch.stack([transforms.functional.rotate(img, 180) for img in images])
                # augmented_270 = torch.stack([transforms.functional.rotate(img, 270) for img in images])

                # combined_images = torch.cat([images, augmented_90, augmented_180, augmented_270], dim=0)
                # combined_labels = torch.cat([labels, labels, labels, labels], dim=0)

                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)
                # outputs = model(combined_images)
                loss = args.criterion(outputs, labels)
                # loss = args.criterion(outputs, combined_labels)
                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Accumulate loss and accuracy
                train_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                # correct += (preds == combined_labels).sum().item()
                total += labels.size(0)
                # total += combined_labels.size(0)
                
                #-----for one hot-----------
                # preds = torch.argmax(outputs, dim=1)
                # targets = torch.argmax(labels, dim=1)  # Convert one-hot back to indices
                # correct += (preds == targets).sum().item()
                # total += targets.size(0)



                # Update progress bar with metrics
                pbar.set_postfix({'Train Loss': train_loss / len(train_loader), 'Accuracy': correct / total})
                # pbar.set_postfix({'Train Loss': train_loss / (4*len(train_loader)), 'Accuracy': correct / total})

        train_acc = correct / total
        train_loss /= len(train_loader)
        # train_loss /= (4*len(train_loader))
        if args.WANDB_TRACKING:
            wandb.log({'Train Loss': train_loss, 'Train Accuracy': train_acc, 'Epoch': epoch})

        # Validation Phase
        valid_loss, correct, total = 0.0, 0, 0
        model.eval()

        with tqdm(valid_loader, desc=f"Validation Epoch {epoch}") as pbar:
            with torch.no_grad():
                for batch in pbar:
                    images, labels = batch["image"].to(args.device), batch["label"].to(args.device)

                    # Forward pass
                    outputs = model(images)
                    loss = args.criterion(outputs, labels)

                    # Accumulate loss and accuracy
                    valid_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                    #------------for one hot-----
                    # preds = torch.argmax(outputs, dim=1)
                    # targets = torch.argmax(labels, dim=1)
                    # correct += (preds == targets).sum().item()
                    # total += targets.size(0)

                    # Update progress bar with metrics
                    pbar.set_postfix({'Valid Loss': valid_loss / len(valid_loader), 'Accuracy': correct / total})

        valid_acc = correct / total
        valid_loss /= len(valid_loader)

        if args.WANDB_TRACKING:
            wandb.log({'Valid Loss': valid_loss, 'Valid Accuracy': valid_acc, 'Epoch': epoch})

        # Check if model improved
        if valid_loss < MIN_loss or max_val<valid_acc:
            if valid_loss < MIN_loss:
                MIN_loss = valid_loss
            if max_val<valid_acc:
                max_val = valid_acc
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            print(f"Model improved: Best VAL LOSS: {MIN_loss:.6f} at epoch {best_epoch}\n")

            # Save the best model
            checkpoint_path = os.path.join(args.checkpoint_path, f'best_model_epoch_{best_epoch}_val_acc_{valid_acc}.pth')
            torch.save(model, checkpoint_path)
            if args.WANDB_TRACKING:
                wandb.save(checkpoint_path)

        if args.use_scheduler:
            scheduler.step(valid_loss)

        # Save model snapshot every few epochs
        if epoch % args.snapshot == 0:
            checkpoint_path = os.path.join(args.checkpoint_path, f'model_epoch_{epoch}_100_data.pth')
            torch.save(model, checkpoint_path)

    if args.WANDB_TRACKING:
        wandb.finish()

    return model, best_model


import matplotlib.pyplot as plt
import os
plt.switch_backend('agg')
# Function to visualize augmented images
def save_augmented_images(dataset, save_path, num_samples=5):
    os.makedirs(save_path, exist_ok=True)
    for i, sample in enumerate(dataset):
        if i >= num_samples:  # Limit the number of images to save
            break
        image = sample["image"]
        image = image.permute(1, 2, 0)  # Convert to HWC
        image = image * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])  # De-normalize
        image = image.clamp(0, 1)  # Clamp values to valid range
        
        plt.imsave(os.path.join(save_path, f"augmented_image_{i}.png"), image.numpy())




if __name__ == "__main__":
    args = get_options()
    torch.cuda.empty_cache()
    # Load saved datasets
    train_dataset = torch.load(args.train_dataset_path)
    val_dataset = torch.load(args.val_dataset_path)
    test_dataset = torch.load(args.test_dataset_path)

    train_loader, val_loader, test_loader = create_loaders(train_dataset, val_dataset, test_dataset)
    
    # After creating train_dataset
    save_augmented_images(train_dataset, save_path="/data/talkoz/Image_Style_Transfer/Classification_Task_Aviv/augmented_check/")
    
    model = StyleClassifier(num_classes=13)
    train_model(model=model, train_loader=train_loader, valid_loader=val_loader, args=args)
