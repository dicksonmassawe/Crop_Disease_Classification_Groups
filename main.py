# Import Libraries
import os
import tqdm
import torch
import random
import numpy as np
import seaborn as sns
from tqdm.auto import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors

# Local module imports
from models import *
from data_loader import CropDataLoader

# Set Seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set the seed for reproducibility
SEED = 42
set_seed(SEED)

# Device used Spec
# Check for device
if torch.cuda.is_available():
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_capability = torch.cuda.get_device_capability(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert bytes to GB
    print(f"Using device: {device}")
    print(f"GPU Name: {gpu_name}")
    print(f"GPU Compute Capability: {gpu_capability[0]}.{gpu_capability[1]}")
    print(f"GPU Total Memory: {gpu_memory:.2f} GB")
else:
    device = torch.device('cpu')
    print("Using device: CPU")

# Directory containing class folders with images
base_image_dir = "data"

# Parameter used
batch_size = 32           # Number of samples per batch
epochs = 100              # Total number of training epochs
learning_rate = 3E-5      # Learning rate for the optimizer
momentum = 0.9            # Momentum for the optimizer
pre_fetch = 4             # Number of batches to pre-fetch
workers = os.cpu_count()  # Number of workers (CPU cores)

# Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),       # Normalize image channels to have mean 0.5
        std=(0.5, 0.5, 0.5)         # Normalize image channels to have std 0.5
    ),
])


# Function to train the model on each folder
def train_on_folder(folder_name):
    # Create a directory to store the results of this folder's training
    result_dir = f"results/{folder_name}"
    os.makedirs(result_dir, exist_ok=True)

    # Load Data
    data_loader = CropDataLoader(
        root_dir=os.path.join(base_image_dir, folder_name),  # Path for the specific folder
        transform=transform,  # Apply the defined transformations
        batch_size=batch_size,  # Number of samples per batch
        workers=workers,  # Use the number of available CPU cores
        pre_fetch=pre_fetch  # Number of batches to pre-fetch
    )

    # Prepare the data using the data loader
    train_loader, val_loader, classes = data_loader.prepare_data()

    # Check if classes are found, otherwise raise an error
    if not classes:
        raise ValueError(
            f"No classes found in folder '{folder_name}'. Check your dataset directory and loader implementation.")
    no_classes = len(classes)

    # Print classes
    print(f"Training on folder: {folder_name}")
    print(f"Number of classes: {no_classes}")
    print("Classes:", classes)

    # Initialize the model and move it to the selected device
    model = Model3(num_classes=no_classes)  # Use the current best model
    model = model.to(device)  # Move the model to GPU (if available) or CPU

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification
    optimizer = optim.SGD(
        model.parameters(),  # Parameters to optimize
        lr=learning_rate,  # Learning rate
        momentum=momentum  # Momentum for optimization
    )

    # Training settings
    batches = len(train_loader)  # Total number of batches in the training data

    # Lists to store training and validation metrics
    train_losses = []  # Training loss for each epoch
    val_losses = []  # Validation loss for each epoch
    train_accuracies = []  # Training accuracy for each epoch
    val_accuracies = []  # Validation accuracy for each epoch

    # Training Function
    def training():
        for epoch in range(epochs):
            # Training Phase
            model.train()
            progress = tqdm(enumerate(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", total=batches)
            total_loss = 0
            correct = 0
            total = 0

            for i, (inputs, labels) in progress:
                # Move inputs and labels to the specified device (CPU/GPU)
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                output = model(inputs)

                # Compute loss
                loss = criterion(output, labels)

                # Backward pass and optimization step
                loss.backward()
                optimizer.step()

                # Update metrics
                current_loss = loss.item()
                total_loss += current_loss
                _, predicted = torch.max(output.data, 1)  # Get predictions
                total += labels.size(0)  # Total samples
                correct += (predicted == labels).sum().item()  # Correct predictions

                # Update progress bar description
                progress.set_description("Loss: {:.4f}".format(total_loss / (i + 1)))

            # Calculate epoch metrics
            epoch_loss = total_loss / batches
            epoch_accuracy = (correct / total) * 100

            # Store training metrics
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%")

            # Validation Phase
            model.eval()
            val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():  # Disable gradient computation for validation
                for inputs, labels in val_loader:
                    # Move inputs and labels to the specified device
                    inputs, labels = inputs.to(device), labels.to(device)

                    # Forward pass
                    outputs = model(inputs)

                    # Compute loss
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    # Update metrics
                    _, predicted = torch.max(outputs.data, 1)  # Get predictions
                    total += labels.size(0)  # Total samples
                    correct += (predicted == labels).sum().item()  # Correct predictions

            # Calculate validation metrics
            val_loss /= len(val_loader)
            val_accuracy = (correct / total) * 100

            # Store validation metrics
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(f"Epoch {epoch + 1}/{epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Save the trained model weights
        torch.save(model.state_dict(), os.path.join(result_dir, 'model_weights.pth'))
        print(f"Model weights saved to '{result_dir}/model_weights.pth'")

    # Visualize training results
    def visualize_results():
        # Plot training and validation loss
        plt.figure()
        plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
        plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epochs', color='blue')
        plt.ylabel('Loss', color='blue')
        plt.title('Training and Validation Loss', fontweight='bold')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()  # Ensures layout fits properly
        plt.savefig(os.path.join(result_dir, 'loss_plot.png'))  # Save to the specific folder
        plt.close()

        # Plot training and validation accuracy
        plt.figure()
        plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
        plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs', color='blue')
        plt.ylabel('Accuracy (%)', color='blue')
        plt.title('Training and Validation Accuracy', fontweight='bold')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()  # Ensures layout fits properly
        plt.savefig(os.path.join(result_dir, 'accuracy_plot.png'))  # Save to the specific folder
        plt.close()

        # Confusion Matrix (Validation Phase)
        model.eval()
        confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        # Precision, Recall, and F1-Score Calculation (Fix for division by zero)
        epsilon = 1e-10  # Small value to avoid division by zero
        precision = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=0) + epsilon)
        recall = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + epsilon)
        f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

        # Plot Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted', color='blue')
        plt.ylabel('Actual', color='blue')
        plt.xticks(rotation=45, ha='right')
        plt.title('Confusion Matrix', fontweight='bold')
        plt.tight_layout()  # Prevents labels from being cut off
        plt.savefig(os.path.join(result_dir, 'confusion_matrix.png'))  # Save to the specific folder
        plt.close()

        # Plot Precision, Recall, and F1-Score
        plt.figure()
        x = np.arange(len(classes))
        plt.bar(x - 0.2, precision, width=0.2, label='Precision')
        plt.bar(x, recall, width=0.2, label='Recall')
        plt.bar(x + 0.2, f1_score, width=0.2, label='F1-Score')
        plt.xticks(x, classes, rotation=45, ha='right')  # Ensures labels are fully visible
        plt.xlabel('Classes', color='blue')
        plt.ylabel('Metrics', color='blue')
        plt.xticks(rotation=45, ha='right')
        plt.title('Precision, Recall, and F1-Score', fontweight='bold')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()  # Prevents labels from being cut off
        plt.savefig(os.path.join(result_dir, 'metrics.png'))  # Save to the specific folder
        plt.close()

        # Class Distribution
        class_counts = np.sum(confusion_matrix, axis=1)
        plt.figure()
        plt.bar(classes, class_counts)
        plt.xlabel('Classes', color='blue')
        plt.ylabel('Number of Samples', color='blue')
        plt.title('Class Distribution in Validation Set', fontweight='bold')
        plt.xticks(rotation=45, ha='right')  # Ensures labels are fully visible
        plt.grid(True)
        plt.tight_layout()  # Prevents labels from being cut off
        plt.savefig(os.path.join(result_dir, 'class_distribution.png'))  # Save to the specific folder
        plt.close()

    def plot_knn_visualization():
        # Extract features and labels from the validation dataset
        features = []
        labels = []

        model.eval()
        with torch.no_grad():
            for inputs, label_batch in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                features.append(outputs.cpu().numpy())  # Collect features
                labels.extend(label_batch.numpy())  # Collect labels

        features = np.vstack(features)  # Stack feature batches
        labels = np.array(labels)

        # Reduce dimensions to 2D using t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        reduced_features = tsne.fit_transform(features)

        # k-NN algorithm
        knn = NearestNeighbors(n_neighbors=5)
        knn.fit(reduced_features)

        # Generate distinct colors for each class
        cmap = plt.colormaps['tab10']  # Use a colormap (e.g., 'tab10')

        # Define a list of marker styles (you can add more if needed)
        markers = ['o', 's', 'D', '^', 'v', 'p', '*', '+', 'x', '|']  # Add more markers if needed

        # Plot t-SNE visualization
        plt.figure(figsize=(10, 8))
        for class_index, class_name in enumerate(classes):
            class_points = reduced_features[labels == class_index]
            plt.scatter(
                class_points[:, 0],
                class_points[:, 1],
                label=class_name,
                alpha=0.6,
                color=cmap(class_index),  # Assign a unique color to each class
                marker=markers[class_index % len(markers)]  # Cycle through markers for each class
            )

        plt.title("k-Nearest Neighbors Visualization of Classes", fontweight='bold')
        plt.xlabel("t-SNE Feature 1", color='blue')
        plt.ylabel("t-SNE Feature 2", color='blue')

        # Move the legend outside the plot
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Classes")

        plt.grid(True)
        plt.tight_layout()

        # Save to the specific folder
        plt.savefig(os.path.join(result_dir, "knn_visualization.png"))
        plt.close()

    # Start training and visualization
    training()
    visualize_results()
    plot_knn_visualization()

    # Clear GPU memory before moving on to the next training
    del model
    torch.cuda.empty_cache()


# Main function to train models on all folders
def train_on_all_folders():
    # Get all subfolders (classes) in the base directory
    folders = [folder for folder in os.listdir(base_image_dir) if os.path.isdir(os.path.join(base_image_dir, folder))]

    for folder in folders:
        train_on_folder(folder)


if __name__ == "__main__":
    # Start training on all folders
    train_on_all_folders()
