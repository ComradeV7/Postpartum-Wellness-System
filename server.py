import flwr as fl
import sys

def main():
    """
    Main function to start the Federated Learning server.
    """
    # 1. Define the Strategy (Federated Averaging - FedAvg)
    # We configure it to wait for at least 2 clients to be available.
    # It will use at least 2 clients for training (fit) in each round.
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,  # Wait for at least 2 clients to connect
        min_fit_clients=2,        # Use at least 2 clients for training
        min_evaluate_clients=2,   # (Optional) Use 2 clients for evaluation
    )

    # 2. Define the Server Configuration
    # We set it to run for 3 rounds of training.
    config = fl.server.ServerConfig(num_rounds=3)

    print("Starting Federated Learning Server...")
    print(f"Waiting for {strategy.min_available_clients} clients to connect...")
    print(f"Will run for {config.num_rounds} rounds.")

    # 3. Start the Server
    # It will listen on all network interfaces (0.0.0.0) on port 8080.
    try:
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=config,
            strategy=strategy
        )
    except Exception as e:
        print(f"\n[ERROR] Could not start server: {e}")
        print("This may be because the port 8080 is already in use.")
        sys.exit(1)

    print("Federated Learning complete.")

if __name__ == "__main__":
    main()
