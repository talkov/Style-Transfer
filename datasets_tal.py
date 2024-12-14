#modified dataset creation 


from datasets import load_dataset
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from config import get_options

# Define preprocessing function in the global scope
def preprocess_function(examples):
    images = [global_transform(image) for image in examples["image"]]
    labels = examples["style"]  # Use "style" as the target label
    return {"image": images, "label": labels}

# Define global transform (fixed, no dynamic creation)
global_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def make_dataset(args):
    # Load the dataset
    dataset = load_dataset("huggan/wikiart", split="train[:40%]")

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
