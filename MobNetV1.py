import pandas as pd
import numpy as np
import torch
from torchvision.datasets import ImageNet
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import time
import os
from torch.optim.lr_scheduler import LambdaLR

data_dir = "/array/data/imagenet/2012"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 256
learning_rate = 0.005
num_epochs = 200
n_workers = 16
weight_decay = 5e-5

train_transform_features = transforms.Compose([
    #transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Loading images...")
#Datasets and loaders
train_dataset = ImageNet(root=data_dir, split="train", transform=train_transform_features)
print("Train loaded.", len(train_dataset), "examples found.")
val_dataset = ImageNet(root=data_dir, split="val", transform=train_transform_features)
print("Val loaded.", len(val_dataset), "examples found.")
test_dataset = val_dataset

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True)
print("Images loaded.")

print("Train loaded.", len(train_dataset), "examples found.")


class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=in_channels)

    def forward(self, x):
        return self.conv(x)


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()

        self.features = nn.Sequential(
            #Initial Conv2D
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 224x224 -> 112x112
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            #Depthwise + Pointwise
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32),  # Depthwise, 112x112 -> 112x112
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),  # Pointwise, 112x112 -> 112x112

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64),  # Depthwise, 112x112 -> 56x56
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),  # Pointwise, 56x56 -> 56x56

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128),  # Depthwise, 56x56 -> 56x56
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),  # Pointwise, 56x56 -> 56x56

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128),  # Depthwise, 56x56 -> 28x28
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),  # Pointwise, 28x28 -> 28x28

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256),  # Depthwise, 28x28 -> 28x28
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),  # Pointwise, 28x28 -> 28x28

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, groups=256),  # Depthwise, 28x28 -> 14x14
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0),  # Pointwise, 14x14 -> 14x14

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512),  # Depthwise, 14x14 -> 14x14
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),  # Pointwise, 14x14 -> 14x14

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512),  # Depthwise, 14x14 -> 14x14
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),  # Pointwise, 14x14 -> 14x14

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512),  # Depthwise, 14x14 -> 14x14
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),  # Pointwise, 14x14 -> 14x14

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512),  # Depthwise, 14x14 -> 14x14
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),  # Pointwise, 14x14 -> 14x14

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512),  # Depthwise, 14x14 -> 14x14
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),  # Pointwise, 14x14 -> 14x14

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, groups=512),  # Depthwise, 14x14 -> 7x7
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0),  # Pointwise, 7x7 -> 7x7

            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1, groups=1024),  # Depthwise, 7x7 -> 7x7
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),  # Pointwise, 7x7 -> 7x7
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  #Global Average Pooling
            nn.Flatten(),
            nn.Linear(1024, num_classes)  #Fully Connected layer
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def calculate_loss_and_accuracy(loader, model, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    avg_loss = running_loss / total
    accuracy = correct / total
    model.train()
    return avg_loss, accuracy

def calculate_total_parameters(model):
    """Calculate the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())

#Save checkpoint
def save_checkpoint(epoch, model, optimizer, scheduler, val_loss, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss
    }
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)

    if is_best:
        best_model_path = f"best_{model_name}_imagenet.pth"
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model updated at epoch {epoch} with val_loss: {val_loss:.4f}")

model = MobileNetV1(num_classes=1000).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, momentum=0.8, weight_decay=weight_decay)
early_stop_patience = 5
best_val_loss = float('inf')
early_stop_counter = 0
#Polynomial decay function
def polynomial_decay(epoch, total_epochs, initial_lr=0.045, power=1.3):
    return initial_lr * (1 - epoch / total_epochs) ** power

# scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: polynomial_decay(epoch, num_epochs))
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: polynomial_decay(epoch, num_epochs))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# # GridSearch
# weight_decay_values = [4e-5, 5e-5]
# power_values = [1.1, 1.2, 1.3]
# learning_rate_values = [0.0040, 0.0045, 0.0050]
# momentum_values = [0.8, 0.9]
#
# best_loss = float("inf")
# best_params = None
#
# for weight_decay, power, learning_rate, momentum in product(weight_decay_values, power_values, learning_rate_values, momentum_values):
#     print(f"Testing: weight_decay={weight_decay}, power={power}, learning_rate={learning_rate}, momentum={momentum}")
#
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
#
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / num_epochs) ** power)
#
#     loss_fn = nn.CrossEntropyLoss()
#
#     # Train for 1 epoch
#     model.train()
#     total_loss = 0.0
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = loss_fn(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#
#     avg_loss = total_loss / len(train_loader)
#     print(f"Final Avg Loss for this combination: {avg_loss:.4f}")
#
#     if avg_loss < best_loss:
#         best_loss = avg_loss
#         best_params = (weight_decay, power, learning_rate, momentum)
#
# print(f"Best Params: weight_decay={best_params[0]}, power={best_params[1]}, learning_rate={best_params[2]}, momentum={best_params[3]}")

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure directory exists

checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")])

#Checkpoints check
if checkpoint_files:
    #latest_checkpoint = checkpoint_files[-1]
    latest_checkpoint = "checkpoint_epoch_120.pth"
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

    #Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['val_loss']

    print(f"Resuming training from {latest_checkpoint} (Epoch {start_epoch})")
else:
    print("No checkpoint found, starting training from scratch.")
    start_epoch = 0
    best_val_loss = float('inf')

#TensorBoard writer
timestamp = time.strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir=f"runs_mobnet2")

#Training loop
for epoch in range(start_epoch, num_epochs):
    print(f"Running epoch {epoch}")
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_train / total_train
    val_loss, val_accuracy = calculate_loss_and_accuracy(val_loader, model, criterion)

    writer.add_scalar('Loss/Train', train_loss, global_step=epoch)
    writer.add_scalar('Loss/Validation', val_loss, global_step=epoch)
    writer.add_scalar('Accuracy/Train', train_accuracy, global_step=epoch)
    writer.add_scalar('Accuracy/Validation', val_accuracy, global_step=epoch)
    writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], global_step=epoch)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
    )

    scheduler.step()

    #Early stop
    if val_loss < best_val_loss :
        best_val_loss = val_loss
        early_stop_counter = 0
        save_checkpoint(epoch, model, optimizer, scheduler, val_loss, is_best=True)  # Save best model
    else:
        early_stop_counter += 1
        save_checkpoint(epoch, model, optimizer, scheduler, val_loss, is_best=False)  # Save regular checkpoint

    if early_stop_counter >= early_stop_patience:
        print("Early stopping triggered. Stopping training.")
        break

#Load the best model
model.load_state_dict(torch.load(f"best_{model_name}_imagenet.pth"))

#Final test loss and accuracy
test_loss, test_accuracy = calculate_loss_and_accuracy(test_loader, model, criterion)
hparams = {
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'num_epochs': num_epochs,
    "model": model_name
}
metrics = {
    'hparam/test_loss': test_loss,
    'hparam/test_accuracy': test_accuracy
}
writer.add_hparams(hparams, metrics)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

writer.close()

