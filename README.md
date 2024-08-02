# Federated Learning with MobileNetV2 using Flower Framework

## Summary
This project demonstrates the implementation of a Federated Learning (FL) setting using the Flower framework and PyTorch. The Large-scale CelebFaces Attributes (CelebA) dataset is used for this purpose. The goal is to train a MobileNetV2 model in a federated manner with 50 clients and compare the model's performance across different demographic groups in the CelebA dataset.

## Method
### 1. Data Splitting
We split the data into two distributions to mimic real-world scenarios:
- **IID distribution**: Data is evenly distributed among clients.
- **Non-IID distribution**: Data is unevenly distributed to represent more realistic and challenging conditions.

### 2. Training and Testing Datasets
The CelebA dataset is preprocessed to create clearly defined training and testing datasets, including class labels.

### 3. Model Preparation
- Load a pre-trained MobileNetV2 model.
- Freeze the feature extractor layers to retain learned features.
- Add a new classifier head and train it from scratch.

### 4. Federated Learning Training
Setup the Flower server and clients to perform federated learning training with 50 clients over at least 10 FL rounds.

### 5. Performance Reporting
Report the training and testing performance using metrics like accuracy, precision, recall, and F1-score.

### 6. Demographic Performance Comparison
Analyze and compare the model's performance across different demographic groups in the CelebA dataset.

## Implementation Steps
### 1. Data Preparation
- **Download the CelebA dataset**: Ensure the dataset is available and properly formatted.
- **Data Splitter**: Implement a script to split data into IID and non-IID distributions.
  ```python
  # Example code snippet for data splitting
  def split_data_iid(data, num_clients):
      # Split data into IID distribution
      pass

  def split_data_non_iid(data, num_clients):
      # Split data into non-IID distribution
      pass
  ```
- **Define Datasets**: Create PyTorch datasets for training and testing.

### 2. Model Setup
- **Load Pre-trained MobileNetV2**: Use torchvision or similar library to load the pre-trained model.
  ```python
  from torchvision.models import mobilenet_v2

  model = mobilenet_v2(pretrained=True)
  ```
- **Freeze Feature Extractor**: Freeze all layers except the classifier head.
  ```python
  for param in model.features.parameters():
      param.requires_grad = False
  ```
- **Add Classifier Head**: Define a new classifier head and attach it to the model.
  ```python
  model.classifier = nn.Sequential(
      nn.Dropout(p=0.2, inplace=False),
      nn.Linear(in_features=model.last_channel, out_features=num_classes, bias=True)
  )
  ```

### 3. Federated Learning
- **Flower Server and Clients**: Setup the Flower server and implement client logic.
  ```python
  # Server setup
  import flwr as fl

  def get_evaluate_fn():
      # Define evaluation function
      pass

  fl.server.start_server("0.0.0.0:8080", config={"num_rounds": 10}, strategy=strategy)

  # Client setup
  class FlowerClient(fl.client.NumPyClient):
      def get_parameters(self):
          # Return model parameters
          pass

      def fit(self, parameters, config):
          # Train model
          pass

      def evaluate(self, parameters, config):
          # Evaluate model
          pass
  ```
- **Federated Learning Rounds**: Execute the federated learning process with multiple clients.
  ```python
  fl.client.start_numpy_client("0.0.0.0:8080", client=FlowerClient())
  ```

## Hyperparameter Tuning
- **Learning Rate**: Adjusted to balance convergence speed and stability.
- **Batch Size**: Selected based on memory constraints and convergence behavior.
- **Number of Rounds**: Set to ensure sufficient training while avoiding overfitting.

## Results
- **Performance Metrics**: Detailed metrics are presented in figures and diagrams.
- **IID vs Non-IID**: Comparison of model performance between IID and non-IID distributions.
- **Demographic Analysis**: Insights on model performance across different demographic groups.

## Assumptions and Parameters
- **Assumed Equal Client Capabilities**: All clients have similar computational and communication capabilities.
- **Documented Parameters**: Learning rate, batch size, number of rounds, and other parameters are documented in the code.

## Usage
1. **Setup Environment**: Install required dependencies using `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```

2. **Flower Server**: Run the Flower server:
```bash
python server.py
```

3. **Flower Client**: Run 50 instances of the Flower client (preferably on different machines or in different terminal windows):
```bash
python client.py
```

## Contact
For any questions or further information, please contact [your email].

```

This more detailed `README.md` file provides step-by-step instructions and explanations for each part of the project, ensuring that users can follow along and understand the implementation and results.
```