import torch
import torchvision.transforms as transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader, Subset, random_split
import torch.nn as nn
import torch.optim as optim
import flwr as fl
import numpy as np
import warnings
import argparse

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-c_id", "--client_id", type=int, required=True, default=0, help = "Client Index")
args = parser.parse_args()

warnings.filterwarnings("ignore", category=UserWarning)

# Define MobileNetV2 model with a new classifier head
class MobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2, self).__init__()
        mobilenet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.feature_extractor = mobilenet.features
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

# Prepare CelebA dataset
def load_data():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = CelebA(root="data", split="all", transform=transform, download=True)
    return dataset

# Split dataset IID
def iid_split(dataset, num_clients):
    indices = np.random.permutation(len(dataset))
    split_indices = np.array_split(indices, num_clients)
    return [Subset(dataset, idx) for idx in split_indices]

# Split dataset non-IID
def non_iid_split(dataset, num_clients):
    targets = dataset.attr[:, 20]  # Using the 'Smiling' attribute for example
    clients_data = []
    for client_id in range(num_clients):
        client_indices = np.where(targets == (client_id % 2))[0]  # Alternate attribute for simplicity
        client_data = Subset(dataset, client_indices)
        clients_data.append(client_data)
    return clients_data

# Create Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader, device):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.classifier.parameters(), lr=0.001, momentum=0.9)

    def get_parameters(self, config):
        return [val.cpu().detach().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        for val, param in zip(parameters, self.model.parameters()):
            param.data = torch.tensor(val).to(self.device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):  # single epoch for demo
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct = 0, 0
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss += self.criterion(outputs, labels).item()
                correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
        return float(loss) / len(self.testloader.dataset), len(self.testloader.dataset), {"accuracy": correct / len(self.testloader.dataset)}

def main():
    # Load and preprocess data
    dataset = load_data()
    num_clients = 50

    # Split dataset (choose IID or non-IID)
    iid = True
    if iid:
        client_datasets = iid_split(dataset, num_clients)
    else:
        client_datasets = non_iid_split(dataset, num_clients)

    # Select one client's dataset for this instance
    client_id = args.client_id  # Change this ID for different clients
    train_size = int(0.8 * len(client_datasets[client_id])) #80-20 Train Test split
    test_size = len(client_datasets[client_id]) - train_size
    train_dataset, test_dataset = random_split(client_datasets[client_id], [train_size, test_size])
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MobileNetV2(num_classes=5).to(device)  # Assuming classification across 5 demographics groups

    print("SUCCESS")
    # Start Flower client
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=FlowerClient(model, trainloader, testloader, device))

if __name__ == "__main__":
    main()
