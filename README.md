# Adversarial Training Service for CNN Models

This project provides a web-based service to enhance the robustness of CNN models using adversarial training techniques. Users can select different security levels (low, medium, or high) to test and improve their model's resilience against adversarial attacks.

---

## Features

- **Adversarial Training**: Incorporates adversarial attacks during training to improve model robustness.
- **Security Levels**:
  - **Low**: FGSM (Fast Gradient Sign Method) attack with a small epsilon value.
  - **Medium**: PGD (Projected Gradient Descent) attack with moderate epsilon and iterations.
  - **High**: CW (Carlini-Wagner) attack, a stronger and more sophisticated adversarial method.
- **Web Interface**: Flask-based interface for user interaction to upload images, select security levels, and train the model.
- **Evaluation**: Displays model accuracy after adversarial training.

---

## Workflow

The workflow of the application consists of the following steps:

1. **Flask Setup**:
   - Flask is used to create a web interface.
   - PyTorch is utilized for model training, and `torchattacks` is used for generating adversarial examples.

2. **Define the CNN Model**:
   - A simple CNN model is implemented using PyTorch.
   - The model includes a convolutional layer, ReLU activation, max pooling, and a fully connected layer for classifying the MNIST dataset.

3. **Load MNIST Dataset**:
   - The MNIST dataset is loaded using `torchvision.datasets.MNIST`.
   - Data is transformed into tensors and normalized, with training data shuffled for randomness.

4. **Adversarial Training**:
   - Based on the selected security level, adversarial attacks are applied:
     - **Low Security**: FGSM with a small epsilon (0.1).
     - **Medium Security**: PGD with moderate epsilon and iterations.
     - **High Security**: CW attack for stronger adversarial training.
   - Adversarial examples are generated during training and used for backpropagation.

5. **Model Evaluation**:
   - The trained model is evaluated using the test dataset.
   - Accuracy is calculated by comparing predicted labels with true labels.

6. **Flask Application**:
   - Users can interact via a web interface to:
     - Upload images.
     - Select the desired security level.
     - Train the model and view the accuracy.

---

## Installation

Follow these steps to set up and run the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/adversarial-training-service.git
   cd adversarial-training-service
2. Install the required dependencies
   ```bash
   pip install -r requirements.txt
3. Run the flask application
   ```bash
   python adversarial_training_service.py

---

## License
This project is licensed under the MIT License.


