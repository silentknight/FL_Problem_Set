import torch
import torchvision.transforms as transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
    print("Data loaded")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = CelebA(root="data", split="all", transform=transform, download=True)
    return dataset

# Split dataset IID
def iid_split(dataset, num_clients):
    """
    Splits the dataset into a IID distribution among clients.

    Parameters:
    - dataset: The dataset to be split.
    - num_clients: The number of clients.

    Returns:
    - A dictionary where keys are client indices and values are lists of dataset indices.
    """

    print("Creating IID data")
    indices = np.random.permutation(len(dataset))
    split_indices = np.array_split(indices, num_clients)

    return [Subset(dataset, idx) for idx in split_indices]

# Split dataset non-IID
def non_iid_split(dataset, num_clients):
    """
    Splits the dataset into a non-IID distribution among clients.
    
    Parameters:
    - dataset: The dataset to be split.
    - num_clients: The number of clients.
    
    Returns:
    - A dictionary where keys are client indices and values are lists of dataset indices.
    """

    print("Creating Non-IID splits")
    targets = dataset.attr[:, 20]  # Using the 'Smiling' attribute for example
    clients_data = []
    for client_id in range(num_clients):
        client_indices = np.where(targets == (client_id % 2))[0]  # Alternate attribute for simplicity
        client_data = Subset(dataset, client_indices)
        clients_data.append(client_data)
    return clients_data

# def non_iid_split(dataset, num_clients, num_shards_per_client=2):
#     """
#     Splits the dataset into a non-IID distribution among clients.
    
#     Parameters:
#     - dataset: The dataset to be split.
#     - num_clients: The number of clients.
#     - num_shards_per_client: The number of shards per client (default: 2).
    
#     Returns:
#     - A dictionary where keys are client indices and values are lists of dataset indices.
#     """
#     num_shards = num_clients * num_shards_per_client
#     num_samples_per_shard = len(dataset) // num_shards

#     # Sort the dataset by labels (assuming targets are available)
#     indices = np.argsort(dataset.attr[:, 20].numpy())  # Example: using the 'Smiling' attribute
    
#     # Split indices into shards
#     shards = [indices[i * num_samples_per_shard:(i + 1) * num_samples_per_shard] for i in range(num_shards)]
    
#     # Assign shards to clients
#     client_data = {i: np.concatenate(shards[i * num_shards_per_client:(i + 1) * num_shards_per_client])
#                    for i in range(num_clients)}

#     print(client_data)    
#     return client_data

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
        print("Getting model parameters")
        return [val.cpu().detach().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        print("Setting model parameters")
        for val, param in zip(parameters, self.model.parameters()):
            param.data = torch.tensor(val).to(self.device)

    def fit(self, parameters, config):
        print("Starting training round")
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):  # single epoch for demo
            running_loss = 0.0
            for i, (inputs, labels) in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 10 == 9:  # Print every 10 batches
                    print(f"Batch {i+1}: Loss = {running_loss / 10:.4f}")
                    running_loss = 0.0

        print("Training round completed")
        return self.get_parameters(), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print("Starting evaluation")
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0
        correct = 0,
        x_true = []
        y_true = []

        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss += self.criterion(outputs, labels).item()
                correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()

                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()

                y_true.extend(labels.numpy())
                y_pred.extend(pred.numpy())

        accuracy = correct / correct / len(self.testloader.dataset)

        accuracy_s = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        print(f"Evaluation completed: Loss = {loss / len(testloader):.4f}, Accuracy = {accuracy:.4f}")

        return float(loss) / len(self.testloader.dataset), len(self.testloader.dataset), {"accuracy": accuracy}, {"accuracy_s": accuracy_s}, {"precision": precision}, {"recall": recall}, {"f1": f1}

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
    model = MobileNetV2(num_classes=2).to(device)

    print("SUCCESS")
    # Start Flower client
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=FlowerClient(model, trainloader, testloader, device))

if __name__ == "__main__":
    main()
