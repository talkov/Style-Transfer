from tqdm import tqdm  # Import tqdm for progress tracking
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch
# import wandb
import os
from config import get_options
from model import StyleClassifier
from dataset import create_loaders, preprocess_function, make_dataset

def train_model(model, train_loader, valid_loader, args):
    # Load model if specified
    if args.load_model:
        model = torch.load(args.load_model_path)
    model = model.to(args.device)

    best_model = copy.deepcopy(model)
    MIN_loss = np.inf
    best_epoch = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.scheduler_factor, patience=args.scheduler_patience, verbose=True)

    for epoch in range(1, args.num_epochs + 1):
        print(f"Epoch {epoch}/{args.num_epochs}\n")

        # Training Phase
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        with tqdm(train_loader, desc=f"Training Epoch {epoch}") as pbar:
            for batch in pbar:
                images, labels = batch["image"].to(args.device), batch["label"].to(args.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)
                loss = args.criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Accumulate loss and accuracy
                train_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                # Update progress bar with metrics
                pbar.set_postfix({'Train Loss': train_loss / len(train_loader), 'Accuracy': correct / total})

        train_acc = correct / total
        train_loss /= len(train_loader)

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

                    # Update progress bar with metrics
                    pbar.set_postfix({'Valid Loss': valid_loss / len(valid_loader), 'Accuracy': correct / total})

        valid_acc = correct / total
        valid_loss /= len(valid_loader)

        # Check if model improved
        if valid_loss < MIN_loss:
            MIN_loss = valid_loss
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            print(f"Model improved: Best VAL LOSS: {MIN_loss:.6f} at epoch {best_epoch}\n")

            # Save the best model
            checkpoint_path = os.path.join(args.checkpoint_path, 'best_model.pth')
            torch.save(model, checkpoint_path)

        if args.use_scheduler:
            scheduler.step(valid_loss)

        # Save model snapshot every few epochs
        if epoch % args.snapshot == 0:
            checkpoint_path = os.path.join(args.checkpoint_path, f'model_epoch_{epoch}.pth')
            torch.save(model, checkpoint_path)

    return model, best_model

if __name__ == "__main__":
    args = get_options()

    # Load saved datasets
    train_dataset = torch.load(args.train_dataset_path)
    val_dataset = torch.load(args.val_dataset_path)
    test_dataset = torch.load(args.test_dataset_path)

    train_loader, val_loader, test_loader = create_loaders(train_dataset, val_dataset, test_dataset)

    model = StyleClassifier(num_classes=27)
    train_model(model=model, train_loader=train_loader, valid_loader=val_loader, args=args)
