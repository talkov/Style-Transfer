
from datasets import load_dataset
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from collections import defaultdict
from config import get_options
from datasets import load_from_disk

# Define the number of classes
NUM_CLASSES = 13
MAX_SAMPLES_PER_CLASS = 3000  # Limit per class

# Define preprocessing function in the global scope
def preprocess_function(examples):
    images = [global_transform(image) for image in examples["image"]]
    labels = examples["style"]  # Use "style" as the target label
    # Convert labels to one-hot encoding
    one_hot_labels = [F.one_hot(torch.tensor(label), num_classes=NUM_CLASSES).tolist() for label in labels]
    return {"image": images, "label": one_hot_labels}

# Define global transform (fixed, no dynamic creation)
global_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def limit_samples_per_class(dataset, max_samples_per_class):
    # Create a dictionary to keep track of counts per class
    class_counts = defaultdict(int)

    # Filter the dataset
    def filter_by_class(example):
        class_label = example["style"]
        if class_counts[class_label] < max_samples_per_class:
            class_counts[class_label] += 1
            return True
        return False

    # Apply filtering
    return dataset.filter(filter_by_class)

def make_dataset(args):
    # Load the dataset
    dataset = load_from_disk("/data/talkoz/Image_Style_Transfer/Classification_Task_Aviv/Datasets/WikiArt/remaped_raw_filterd")

    # Limit samples per class
    dataset = limit_samples_per_class(dataset, MAX_SAMPLES_PER_CLASS)

    # Split dataset into train, validation, and test
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_valid_split = train_test_split['train'].train_test_split(test_size=0.125, seed=42)  # 0.125 * 0.8 = 10%

    train_dataset = train_valid_split["train"]
    val_dataset = train_valid_split["test"]
    test_dataset = train_test_split["test"]

    # Apply preprocessing
    train_dataset = train_dataset.with_transform(preprocess_function)
    val_dataset = val_dataset.with_transform(preprocess_function)
    test_dataset = test_dataset.with_transform(preprocess_function)

    # Save datasets
    torch.save(test_dataset, args.test_dataset_path)
    torch.save(val_dataset, args.val_dataset_path)
    torch.save(train_dataset, args.train_dataset_path)

    return train_dataset, val_dataset, test_dataset

def create_loaders(train_dataset, val_dataset, test_dataset):
    # Create PyTorch DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    args = get_options()

    # Uncomment this to create and save the datasets once
    make_dataset(args)

    # Load saved datasets
    train_dataset = torch.load(args.train_dataset_path)
    val_dataset = torch.load(args.val_dataset_path)
    test_dataset = torch.load(args.test_dataset_path)

    train_loader, val_loader, test_loader = create_loaders(train_dataset, val_dataset, test_dataset)


file_path = "/data/talkoz/Image_Style_Transfer/Classification_Task_Aviv/Datasets/WikiArt/limit_3000_filtered/traindataset_augmented.pth"
print(f"Loading dataset from: {file_path}")
train_data = torch.load(file_path)
print(train_data.shape)
