# PlantDiseaseDetectionFTL
Implementation of federated transfer learning for tomato leaf disease identification.

PROJECT DESCRIPTION:
The project implements federated transfer learning between 2 clients A and B in order to transfer knowledge from the rich source (client A) to the resource-scarce destionation(clinet B).

Components:

CORAL Loss Function (coral.py):
- Implements the Correlation Alignment (CORAL) loss function, which helps in aligning the statistical distributions of feature representations from different domains.
- This is particularly useful for domain adaptation tasks where models are trained on one domain and applied to another.

Federated Learning Server (server.py):
- Sets up a federated learning server using the Flower framework.
- Aggregates model updates from multiple clients using a weighted average strategy.
- Configured to run for a specified number of training rounds.

Centralized Training Script (centralized.py):
- Defines a simple convolutional neural network (CNN) for image classification tasks.
- Sets up a centralized training loop using the CIFAR-10 dataset as an example.
- Incorporates the CORAL loss function to align feature distributions.

Federated Learning Client (client.py):
- Implements a federated learning client using the Flower framework.
- Trains the model locally using client-specific data.
- Communicates with the federated server to send updates and receive aggregated model parameters.
- Configures logging and handles client-specific data loading and model training processes.

