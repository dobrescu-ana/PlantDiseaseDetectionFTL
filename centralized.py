import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms
from coral import coral_loss

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Calculate the input size for the fully connected layer
        self._initialize_fc_layer()

        self.fc1 = nn.Linear(self.fc_input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def _initialize_fc_layer(self):
        # Create input tensor with the correct dimensions
        dummy_input = torch.zeros(1, 3, 256, 256)
        with torch.no_grad():
            dummy_output = self.pool(self.conv1(dummy_input))
            dummy_output = self.pool(self.conv2(dummy_output))
            dummy_output = self.pool2(self.conv3(dummy_output))
        self.fc_input_size = dummy_output.numel()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = x.view(-1, self.fc_input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(model, train_loader, target_loader, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    for _ in range(epochs):
        for (source_images, source_labels), (target_images, _) in zip(train_loader, target_loader):
            optimizer.zero_grad()
            source_outputs = model(source_images.to(DEVICE))
            target_outputs = model(target_images.to(DEVICE))

            classification_loss = criterion(source_outputs, source_labels.to(DEVICE))
            coral_loss_value = coral_loss(source_outputs, target_outputs)

            loss = classification_loss + coral_loss_value
            loss.backward()
            optimizer.step()

def test(model, test_loader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(DEVICE))
            loss += criterion(outputs, labels.to(DEVICE)).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return loss / len(test_loader.dataset), correct / total

def load_data(data_dir):
    train_dir = data_dir + "/train"
    train_dataset = torchvision.datasets.ImageFolder(root=train_dir,
                                                     transform=transforms.Compose([
                                                         transforms.Resize((256, 256)),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                              std=[0.5, 0.5, 0.5])]))
    
    valid_dir = data_dir + "/valid"
    test_dataset = torchvision.datasets.ImageFolder(root=valid_dir,
                                                    transform=transforms.Compose([
                                                        transforms.Resize((256, 256)),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                             std=[0.5, 0.5, 0.5])]))
    
    batch_size = 16

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Use multiple workers to load data in parallel
        pin_memory=True  
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader

def load_model():
    return SimpleNet().to(DEVICE)

if __name__ == "__main__":
    model = load_model()
    train_loader, test_loader = load_data("C:\\Users\\Ana\\Desktop\\B")  
    target_loader = train_loader  # For demonstration, we use the same loader for target data.
    train(model, train_loader, target_loader, 3)  
    loss, accuracy = test(model, test_loader)
    print(f"Loss: {loss:.5f}, Accuracy: {accuracy:.3f}")