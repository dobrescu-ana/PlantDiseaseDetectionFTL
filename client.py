import os
import sys
import logging
from collections import OrderedDict
import flwr as fl
import torch
from torchvision import models, transforms
from centralized import load_data, load_model, train, test
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model

# Define paths for each client - ONLY FOR LOCAL TEST FOR NOW
CLIENT_DATA_DIRS = {
    1: "C:\\Users\\Ana\\Desktop\\B",
    2: "C:\\Users\\Ana\\Desktop\\B"
}

# Parse command-line arguments to get CLIENT_ID
parser = argparse.ArgumentParser(description='Federated Learning Client')
parser.add_argument('--client_id', type=int, required=True, help='Client ID')
args = parser.parse_args()
CLIENT_ID = args.client_id

# Get the data directory for the given CLIENT_ID
data_dir = CLIENT_DATA_DIRS[CLIENT_ID]
model = load_model()
train_loader, test_loader = load_data(data_dir)
target_loader = train_loader  # For demonstration, we use the same loader for target data.

# Flower Client
class FedClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        logging.info("Getting parameters")
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def fit(self, parameters, config):
        logging.info("Starting training")
        set_parameters(model, parameters)
        train(model, train_loader, target_loader, epochs=3)  # Reduced epochs
        logging.info("Finished training")
        return self.get_parameters({}), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        logging.info("Starting evaluation")
        set_parameters(model, parameters)
        loss, accuracy = test(model, test_loader)
        logging.info(f"Finished evaluation with loss: {loss}, accuracy: {accuracy}")
        return float(loss), len(test_loader.dataset), {"accuracy": accuracy}

# Start Flower client
fl.client.start_client(
    server_address="localhost:8081",
    client=FedClient().to_client()
)