import matplotlib.pyplot as plt
import torch

def test_vit(model, test_loader, device, num_classes, class_names):
    """Test the Vision Transformer and calculate per-class accuracy."""
    model.eval()
    correct = 0
    total = 0
    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    # Overall accuracy
    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Per-class accuracy
    per_class_acc = []
    for i in range(num_classes):
        acc = 100 * class_correct[i] / class_total[i]
        per_class_acc.append(acc)
        print(f"Accuracy of {class_names[i]}: {acc:.2f}%")

    # Plot per-class accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, per_class_acc, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy on CIFAR-10 Test Set')
    plt.xticks(rotation=45)
    plt.show()