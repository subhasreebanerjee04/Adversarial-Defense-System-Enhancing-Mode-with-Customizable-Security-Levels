from flask import Flask, request, render_template
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchattacks

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Load a small dataset (MNIST)
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader

# Adversarial training function
def adversarial_training(model, train_loader, security_level, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Select adversarial attack based on security level
    if security_level == "low":
        attack = torchattacks.FGSM(model, eps=0.1)
    elif security_level == "medium":
        attack = torchattacks.PGD(model, eps=0.1, alpha=0.01, steps=40)
    elif security_level == "high":
        attack = torchattacks.CW(model, c=1, kappa=0, steps=100, lr=0.01)
    else:
        raise ValueError("Invalid security level. Choose from 'low', 'medium', 'high'.")

    model.train()
    for epoch in range(1):  # Train for 1 epoch for simplicity
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Generate adversarial examples
            adv_images = attack(images, labels)

            # Forward pass
            outputs = model(adv_images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/1], Loss: {loss.item():.4f}")

# Evaluate the model
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

# Flask application
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    security_level = request.form['security_level']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    train_loader, test_loader = load_data()

    try:
        adversarial_training(model, train_loader, security_level, device)
        accuracy = evaluate_model(model, test_loader, device)
        return f"Model trained with {security_level} security. Accuracy: {accuracy:.2f}%"
    except ValueError as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
