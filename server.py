import flwr as fl
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# # Define evaluation function
# def evaluate(model, data_loader, criterion):
#     model.eval()
#     correct = 0
#     total = 0
#     y_true = []
#     y_pred = []
    
#     with torch.no_grad():
#         for images, labels in data_loader:
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             y_true.extend(labels.numpy())
#             y_pred.extend(predicted.numpy())

#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred, average='macro')
#     recall = recall_score(y_true, y_pred, average='macro')
#     f1 = f1_score(y_true, y_pred, average='macro')
    
#     return accuracy, precision, recall, f1

# # Flower server setup
# def get_evaluate_fn(model, test_loader, criterion):
#     def evaluate(parameters):
#         model.set_parameters(parameters)
#         accuracy, precision, recall, f1 = evaluate(model, test_loader, criterion)
#         return accuracy, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
#     return evaluate


def main():
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        # evaluate_fn=get_evaluate_fn(model, test_loader, criterion),
        fraction_fit=0.5,  # Train on 50 clients (each round)
        fraction_evaluate=0.5,  # Evaluate on 50 clients (each round)
        min_fit_clients=50,
        # min_eval_clients=50,
        min_available_clients=100
     )

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
