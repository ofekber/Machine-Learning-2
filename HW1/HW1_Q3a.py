import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Function to load the data and apply necessary transformations
def load_data(batch_size):
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.ImageFolder(root='hw1_data/dog_emotion/train', transform=transform_train)
    val_dataset = datasets.ImageFolder(root='hw1_data/dog_emotion/val', transform=transform_test)
    test_dataset = datasets.ImageFolder(root='hw1_data/dog_emotion/test', transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# CNN Model
class DogEmotionCNN(nn.Module):
    def __init__(self):
        super(DogEmotionCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 36, kernel_size=2, padding=1),
            nn.BatchNorm2d(36),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(36, 72, kernel_size=6, padding=3),
            nn.BatchNorm2d(72),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(72, 36, kernel_size=6, padding=3),
            nn.BatchNorm2d(36),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(36, 72, kernel_size=6, padding=3),
            nn.BatchNorm2d(72),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(72, 36, kernel_size=6, padding=3),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(36, 72, kernel_size=6, padding=3),
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(72, 36, kernel_size=6, padding=3),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        dummy_input = torch.zeros(1, 3, 128, 128)
        dummy_output = self.conv8(self.conv7(self.conv6(self.conv5(
            self.conv4(self.conv3(self.conv2(self.conv1(dummy_input))))))))
        flattened_size = dummy_output.view(1, -1).size(1)

        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 4)
        self.dropout = nn.Dropout(p=0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return self.logsoftmax(out)