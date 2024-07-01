import flwr as fl
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def weighted_average(metrics):
    logging.info("Aggregating metrics")
    accuracies = [num_example * m["accuracy"] for num_example, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# Define the strategy
strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,
)

# Log server start
logging.info("Starting Flower server")

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8081",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)