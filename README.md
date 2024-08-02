# Federated Learning with MobileNetV2 using Flower Framework

## Summary
This project implements a Federated Learning (FL) setting using the Flower framework and PyTorch. The dataset used is the Large-scale CelebFaces Attributes (CelebA) dataset. The primary goal is to perform FL with 50 clients and compare the performance of the model across different demographic groups in the CelebA dataset.

## Method
### 1. Data Splitting
The data is split into two distributions:
- **IID distribution**: Data is evenly distributed among clients.
- **Non-IID distribution**: Data is unevenly distributed to mimic real-world scenarios.

### 2. Training and Testing Datasets
Clearly define training and testing datasets, including the classes.

### 3. Model Preparation
- Use a pre-trained version of MobileNetV2.
- Freeze the feature extractor.
- Train the classifier head from scratch.

### 4. Federated Learning Training
Execute the federated learning training with 50 clients for at least 10 FL rounds.

### 5. Performance Reporting
Report the training and testing performance using appropriately selected metrics.

### 6. Demographic Performance Comparison
Compare the performance across demographic groups present in the CelebA dataset.

## Main Deliverables
1. **Code**: The complete code is available in this repository.
2. **Discussion and Details**: A detailed discussion of the implementation steps is provided in this README.
3. **Hyperparameter Tuning**: Explanation of hyperparameter tuning and overfitting prevention strategies.
4. **Performance Assessment**: Performance results and discussions are included in the form of figures and diagrams.
5. **Assumptions and Parameters**: Documentation of assumptions made and parameter values used.

## Implementation Steps
### 1. Data Preparation
- Download the CelebA dataset.
- Implement a data splitter to create IID and non-IID distributions.
- Define training and testing datasets.

### 2. Model Setup
- Load the pre-trained MobileNetV2 model.
- Freeze the feature extractor layers.
- Add and train a new classifier head.

### 3. Federated Learning
- Setup the Flower server and clients.
- Implement training and evaluation logic.
- Execute federated learning rounds with 50 clients.

### 4. Performance Reporting
- Calculate accuracy, loss, precision, recall, and F1-score.
- Plot performance metrics for both IID and non-IID distributions.
- Analyze performance across different demographic groups.

## Hyperparameter Tuning
- Learning Rate: Tuned to balance convergence speed and stability.
- Batch Size: Adjusted based on memory constraints and convergence behavior.
- Number of Rounds: Set to ensure sufficient training while avoiding overfitting.

## Results
- Detailed performance metrics are presented in figures and diagrams.
- Comparison of performance between IID and non-IID distributions.
- Analysis of model performance across different demographic groups.

## Assumptions and Parameters
- Assumed equal client capabilities in terms of computation and communication.
- Parameter values such as learning rate, batch size, and number of rounds are documented in the code.

### Inference and Training on Android
- Results of inference measurements using the Android app.
- Discussion on the feasibility and performance of training the model within the app.

## Repository Structure
- `data/`: Contains data preparation scripts and dataset.
- `models/`: Contains model definition and training scripts.
- `fl/`: Contains Flower server and client scripts.
- `results/`: Contains performance metrics and figures.

## Usage
1. **Setup Environment**: Install required dependencies using `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```
2. **Data Preparation**: Run data preparation scripts.
   ```bash
   python data_preparation.py
   ```
3. **Model Training**: Execute the federated learning training.
   ```bash
   python fl_training.py
   ```
4. **Performance Reporting**: Generate performance reports and figures.
   ```bash
   python generate_reports.py
   ```

## Contact
For any questions or further information, please contact [your email].

```

This `README.md` file provides a comprehensive overview of the project, including the method, deliverables, implementation steps, results, and usage instructions. Adjust the paths and file names as needed based on your project structure.
```




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

## Main Deliverables
1. **Code**: The complete code is available in this repository.
2. **Implementation Details**: Detailed explanations of each step in this README.
3. **Hyperparameter Tuning**: Description of hyperparameter tuning strategies and methods to avoid overfitting.
4. **Performance Assessment**: Performance results are presented with figures and diagrams.
5. **Assumptions and Parameters**: Documentation of assumptions and parameter values used.

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

### 4. Performance Reporting
- **Metrics Calculation**: Calculate accuracy, precision, recall, and F1-score.
  ```python
  from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

  def calculate_metrics(y_true, y_pred):
      accuracy = accuracy_score(y_true, y_pred)
      precision = precision_score(y_true, y_pred, average='macro')
      recall = recall_score(y_true, y_pred, average='macro')
      f1 = f1_score(y_true, y_pred, average='macro')
      return accuracy, precision, recall, f1
  ```
- **Performance Visualization**: Plot performance metrics.
  ```python
  import matplotlib.pyplot as plt

  def plot_metrics(metrics):
      # Plot training and testing metrics
      pass
  ```

### 5. Demographic Performance Comparison
- **Demographic Analysis**: Analyze model performance across different demographic groups.
  ```python
  def analyze_demographics(data, model):
      # Perform demographic analysis
      pass
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

## Bonus Section
### Deployment on Android
- **Instructions**: Step-by-step guide for deploying the trained model on an Android app.
- **Android Project Code**: Code included to recreate the app.

### Inference and Training on Android
- **Inference Measurements**: Results of inference measurements using the Android app.
- **Training Performance**: Discussion on the feasibility and performance of training within the app.

## Repository Structure
- `data/`: Contains data preparation scripts and dataset.
- `models/`: Contains model definition and training scripts.
- `fl/`: Contains Flower server and client scripts.
- `results/`: Contains performance metrics and figures.
- `android/`: Contains Android app project for deployment (if applicable).

## Usage
1. **Setup Environment**: Install required dependencies using `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```
2. **Data Preparation**: Run data preparation scripts.
   ```bash
   python data_preparation.py
   ```
3. **Model Training**: Execute the federated learning training.
   ```bash
   python fl_training.py
   ```
4. **Performance Reporting**: Generate performance reports and figures.
   ```bash
   python generate_reports.py
   ```

## Contact
For any questions or further information, please contact [your email].

```

This more detailed `README.md` file provides step-by-step instructions and explanations for each part of the project, ensuring that users can follow along and understand the implementation and results.