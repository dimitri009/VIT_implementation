import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torchvision

def get_data_loaders(image_size=32, batch_size=64, validation_split=0.1):
    """Prepare train, validation, and test data loaders."""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(image_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)

    # Split dataset into training and validation sets
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, dataset.classes