#inference code
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
from model import StyleClassifier
from dataset import create_loaders, preprocess_function, make_dataset
from config import get_options

# Use Agg backend for matplotlib
plt.switch_backend('agg')

def test_model(model, test_loader, args, save_path):
    """
    Tests the trained model on the test dataset, computes metrics, and saves plots.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for test data.
        args: Argument parser containing hyperparameters and settings.
        save_path: Path where plots will be saved.

    Returns:
        None
    """
    model = model.to(args.device)
    model.eval()

    total = 0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch["image"].to(args.device), batch["label"].to(args.device)

            # Forward pass
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Store predictions and labels for analysis
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Count correct predictions
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Save plots
    save_confusion_matrix(all_labels, all_preds, save_path)
    save_tsne(test_loader, model, save_path, args)
    save_class_accuracy_bar(all_labels, all_preds, save_path)

def save_confusion_matrix(true_labels, preds, save_path):
    """
    Saves the confusion matrix as a plot.
    """
    cm = confusion_matrix(true_labels, preds)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(27))
    fig, ax = plt.subplots(figsize=(10, 10))
    display.plot(ax=ax, cmap='viridis', xticks_rotation=45)
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

def save_tsne(test_loader, model, save_path, args):
    """
    Saves a t-SNE visualization of the test dataset.
    """
    embeddings = []
    labels = []

    with torch.no_grad():
        for batch in test_loader:
            images, batch_labels = batch["image"].to(args.device), batch["label"]
            outputs = model(images)
            embeddings.append(outputs.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab20', alpha=0.6)
    plt.colorbar(scatter, label="Classes")
    plt.title('t-SNE Visualization of Test Embeddings')
    plt.savefig(os.path.join(save_path, 'tsne_plot.png'))
    plt.close()

def save_class_accuracy_bar(true_labels, preds, save_path):
    """
    Saves a bar plot showing the accuracy for each class.
    """
    num_classes = 27
    correct_counts = [0] * num_classes
    total_counts = [0] * num_classes

    for true, pred in zip(true_labels, preds):
        total_counts[true] += 1
        if true == pred:
            correct_counts[true] += 1

    class_accuracies = [correct / total if total > 0 else 0 for correct, total in zip(correct_counts, total_counts)]

    plt.figure(figsize=(12, 6))
    plt.bar(range(num_classes), class_accuracies, tick_label=[f"Class {i}" for i in range(num_classes)], color='skyblue')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Class-wise Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'class_accuracy_bar.png'))
    plt.close()

if __name__ == "__main__":
    args = get_options()

    # Specify the save path for plots
    save_path = "/data/talkoz/Image_Style_Transfer/Classification_Task_Aviv/inference_plots"  # Change this to your desired directory
    os.makedirs(save_path, exist_ok=True)

   # Load saved datasets
    train_dataset = torch.load(args.train_dataset_path)
    val_dataset = torch.load(args.val_dataset_path)
    test_dataset = torch.load(args.test_dataset_path)

    train_loader, val_loader, test_loader = create_loaders(train_dataset, val_dataset, test_dataset)

    # Load the trained model
    model = StyleClassifier(num_classes=27)  # Adjust num_classes if needed
    # Load the entire model
    model = torch.load(args.load_model_path, map_location=args.device)
    print(f"Loaded model from {args.load_model_path}")

    # Test the model
    test_model(model, test_loader, args, save_path)

