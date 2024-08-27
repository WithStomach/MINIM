import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import os

from PIL import Image
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.models import resnet50
from torch.optim import Adam
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import pandas as pd


def read_images_from_df(df, root_dir=".", transform=None, index="image_path"):
    images = []
    for image_file in df[index]:
        image_path = os.path.join(root_dir, image_file)
        image = Image.open(image_path)
        if transform == None:
            input_tensor = transforms.ToTensor()(image)
        else:
            input_tensor = transform(image)
        images.append(input_tensor)
    images = torch.stack(images)
    return images


class Classifier(nn.Module):
    def __init__(self, class_num=3):
        super(Classifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 512, 512]
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 256, 256]
            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 256, 256]
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128, 128, 128]
            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 128, 128]
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 64, 64]
            nn.Conv2d(256, 256, 3, 1, 1),  # [256, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 32, 32]
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 32 * 32, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, class_num),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return F.softmax(self.fc(out), dim=1)


def train(model, criterion, optimizer, dataloader, device):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


def validate(model, criterion, dataloader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            _, gt = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == gt).sum().item()

    return running_loss / len(dataloader.dataset), 100.0 * correct / total


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--class_num",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--label_name",
        type=str,
        default="label",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    metadata = pd.read_excel(args.metadata)
    all_images = read_images_from_df(metadata)
    labels = []
    for label in metadata[args.label_name]:
        tensor = torch.zeros((args.class_num,))
        tensor[label - 1] = 1
        labels.append(tensor)
    labels = torch.stack(labels)
    dataset = TensorDataset(all_images, labels)
    trainset, testset = random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = Classifier(class_num=args.class_num)
    num_epochs = 50
    best_model_wts = model.state_dict()
    best_acc = 0.0
    device = args.device
    model = model.to(device)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-5)

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        train_loss = train(model, criterion, optimizer, train_loader, device)

        val_loss, val_acc = validate(model, criterion, test_loader, device)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()

        print(
            "Train Loss: {:.4f} Val Loss: {:.4f} Val Acc: {:.3f}".format(
                train_loss, val_loss, val_acc
            )
        )

    # 保存模型
    torch.save(best_model_wts, "best_tumor_detector.pth")

    print(f"Training complete. The best acc:{best_acc}")
