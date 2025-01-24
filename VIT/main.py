from data import get_data_loaders
import torch.optim as optim
import torch.nn as nn
from VIT import ViT
from train import *
from test import *


def plot_metrics(train_losses, val_accuracies):
    """Plot training loss and validation accuracy."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.show()


def main():
    # Hyperparameters
    image_size = 32
    patch_size = 8
    embedding_dim = 256
    num_blocks = 2
    num_classes = 10
    drop_rate = 0.1
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.001
    weight_decay = 1e-4

    # Get data loaders and class names
    train_loader, val_loader, test_loader, class_names = get_data_loaders(image_size, batch_size)

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))
    model = ViT(image_size=image_size, patch_size=patch_size, embedding_dim=embedding_dim, num_blocks=num_blocks,
                num_classes=num_classes, drop_rate=drop_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

    # Train and validate the model
    train_losses, val_accuracies = train_vit(model, train_loader, val_loader, device, criterion, optimizer, num_epochs)

    # Plot training loss and validation accuracy
    plot_metrics(train_losses, val_accuracies)

    # Test the model
    test_vit(model, test_loader, device, num_classes, class_names)


if __name__ == "__main__":
    main()