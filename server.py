import flwr as fl
import torch
import numpy as np

def main():
    # Define strategy
    class ModelStrategy(fl.server.strategy.FedAvg):
        def aggregate_fit(self, rnd, results, failures):
            aggregated_weights = super().aggregate_fit(rnd, results, failures)
            if aggregated_weights is not None:
                print(f"Saving round {rnd} aggregated_weights...")
                np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
            return aggregated_weights

    strategy = ModelStrategy()

    strategy.fraction_fit=1.0
    strategy.fraction_evaluate=0.0
    strategy.min_fit_clients=1
    strategy.min_evaluate_clients=0
    strategy.min_available_clients=50

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
