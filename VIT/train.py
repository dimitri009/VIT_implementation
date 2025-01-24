import torch



def train_vit(model, train_loader, val_loader, device, criterion, optimizer, num_epochs=10):
    """Train the Vision Transformer and validate after each epoch."""
    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Validation phase
        val_accuracy = validate_vit(model, val_loader, device)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    return train_losses, val_accuracies


def validate_vit(model, val_loader, device):
    """Validate the Vision Transformer."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total