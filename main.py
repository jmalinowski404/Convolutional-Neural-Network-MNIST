import PIL.ImageOps
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import sklearn
import numpy as np
from PIL import Image
from PIL import ImageEnhance

from CNN import *

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset_train = torchvision.datasets.MNIST(root='data', download=False, train=True, transform=transform)
dataset_test = torchvision.datasets.MNIST(root='data', download=False, train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=64, shuffle=False)

network = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
    network.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    network.train()

    for epoch in range(10):
        total_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            with open("train_log.txt", "a") as f:
                if i % 100 == 99:
                    f.write(f"[{epoch + 1}, {i}] loss: {total_loss / 100}\n")
                    total_loss = 0.0
    torch.save(network.state_dict(), "network.pth")

def test():
    network.load_state_dict(torch.load("network.pth", weights_only=True))
    network.eval()
    with torch.no_grad():
        with open("test_log.txt", "a") as f:
            labels_ = []
            predictions_ = []
            for data in test_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = network(inputs)
                _, prediction = torch.max(outputs, 1)

                labels_.append(labels.cpu().numpy())
                predictions_.append(prediction.cpu().numpy())

            all_labels = np.concatenate(labels_)
            all_predictions = np.concatenate(predictions_)

            accuracy = sklearn.metrics.accuracy_score(all_labels, all_predictions)
            confusion_matrix = sklearn.metrics.confusion_matrix(all_labels, all_predictions)
            class_report = sklearn.metrics.classification_report(all_labels, all_predictions)

            f.write(f"Accuracy: {accuracy * 100}%\n")
            f.write(f"Confusion matrix: {confusion_matrix}\n")
            f.write(f"Classification report: {class_report}\n")

def predict():
    network.load_state_dict(torch.load("network.pth", weights_only=True))
    network.to(device)
    network.eval()

    with open("custom_test_log.txt", "w") as f:
        for i in range(10):
            path = f"test_images/{i}.jpg"
            img = Image.open(path)
            grayscale_img = img.convert('L')
            resized_img = grayscale_img.resize((28, 28))
            enhancer = PIL.ImageEnhance.Contrast(resized_img)
            contrasted_img = enhancer.enhance(2.0)
            pointed_img = contrasted_img.point(lambda x: 255 if x > 128 else 0)
            inverted_colors = PIL.ImageOps.invert(pointed_img)
            inverted_colors.save(f"debug_images/debug_{i}.png")
            tensor_image = transform(inverted_colors)
            tensor_image = tensor_image.unsqueeze(0)

            tensor_image = tensor_image.to(device)

            with torch.no_grad():
                output = network(tensor_image)
                _, prediction = torch.max(output, 1)

                inverted_colors.show()
                f.write(f"Plik: {path}\n")
                f.write(f"Predykcja: {prediction.item()}\n\n")

def printMenu():
    print("==== CNN ====")
    print("1. Trenuj")
    print("2. Testuj")
    print("3. Testuj własne zdjęcia")
    print("0. Wyjście")

    choice = int(input("Choice: "))

    match(choice):
        case 1:
            train()
            printMenu()
        case 2:
            test()
            printMenu()
        case 3:
            predict()
            printMenu()
        case 0:
            exit()

if __name__ == "__main__":
    printMenu()