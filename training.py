# Import InMemoryDataset from PyTorch Geometric, a class to handle graph data in memory
from torch_geometric.data import InMemoryDataset

# Import DataLoader from PyTorch Geometric to handle batching and loading of graph data
from torch_geometric.loader import DataLoader

# Import PyTorch for tensor computation
import torch

# Import os for interacting with the operating system, such as file and directory management
import os

# Import DGL (Deep Graph Library) for graph neural networks
import dgl

# Import tqdm to display progress bars in loops
from tqdm import tqdm

# Import NumPy for numerical computations
import numpy as np

# Import metrics from sklearn for evaluating model performance, like confusion matrix and accuracy
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

# Import the MM_IDTarget model from the src.models.MM_IDTarget module
from src.models.MM_IDTarget import MM_IDTarget
    

# Define the SMILES_Protein_Dataset class, which inherits from InMemoryDataset for handling the dataset
class SMILES_Protein_Dataset(InMemoryDataset):
    # Constructor to initialize the dataset class
    def __init__(self, root, raw_dataset=None, processed_data=None, transform=None, pre_transform=None):
        self.root = root  # Set the root directory for the dataset
        self.raw_dataset = raw_dataset  # Path to the raw dataset (if any)
        self.processed_data = processed_data  # Path to the processed dataset
        self.max_smiles_len = 256  # Maximum length of the SMILES strings
        self.smiles_dict_len = 65  # The length of the SMILES dictionary
        super(SMILES_Protein_Dataset, self).__init__(root, transform, pre_transform)  # Call the parent class's constructor
        self.data, self.slices = torch.load(self.processed_paths[0])  # Load the processed dataset from the stored path
        
        
# Define the train function for training the model
def train():
    model.train()  # Set the model to training mode
    total_loss = 0  # Initialize a variable to keep track of the total loss
    for idx, data in loop:  # Loop through the dataset (replace `loop` with actual data loader or iterable)
        # Import necessary PyTorch modules (CrossEntropyLoss is for classification)
        import torch.nn as nn
        loss_fn = nn.CrossEntropyLoss()  # Define the loss function for classification (cross-entropy)
        
        # Forward pass: Pass the data through the model
        out = model(dgl.batch(data.smiles_graph).to(device),  # Pass SMILES graphs to the model
                    dgl.batch(data.pro_graph).to(device),  # Pass protein graphs to the model
                    data.pro_A.to(device),  # Pass protein adjacency matrix to the model
                    data.target.to(device),  # Pass target labels (or features) to the model
                    data.smiles.to(device),  # Pass SMILES representations of the molecules
                    data.ecfp4.to(device))  # Pass the ECFP4 fingerprint representations to the model
        
        # Calculate the loss by comparing the model's output with the true labels
        loss = loss_fn(out, data.y.to(device))  # `data.y` are the ground truth labels for the current batch
        
        # Backpropagation: Reset gradients, compute gradients, and update the model's weights
        optimizer.zero_grad()  # Zero out the gradients from the previous step
        loss.backward()  # Compute the gradients with respect to the loss
        optimizer.step()  # Update the model's parameters based on the gradients
        
        total_loss += loss.item()  # Add the current loss to the total loss
    
    # Return the average loss per batch in the loop
    return total_loss / len(loop)  # `loop` should be replaced with a valid DataLoader or iterable object


# Define the test function for evaluating the model
def test(loader):
    model.eval()  # Set the model to evaluation mode (disables dropout, batch normalization, etc.)
    true_labels = []  # List to store the true labels of the test data
    predicted_labels = []  # List to store the predicted labels from the model
    predicted_probs = []  # List to store the predicted probabilities from the model

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for idx, data in loader:  # Loop through the test data
            optimizer.zero_grad()  # Zero out the gradients (not really needed for testing, but for consistency)
            
            # Forward pass: Pass the test data through the model
            out = model(dgl.batch(data.smiles_graph).to(device),  # Pass SMILES graphs to the model
                        dgl.batch(data.pro_graph).to(device),  # Pass protein graphs to the model
                        data.pro_A.to(device),  # Pass protein adjacency matrix to the model
                        data.target.to(device),  # Pass target labels (or features) to the model
                        data.smiles.to(device),  # Pass SMILES representations of the molecules
                        data.ecfp4.to(device))  # Pass the ECFP4 fingerprint representations to the model
            
            # Get the predicted class labels by taking the argmax of the output (classification task)
            pred1 = out.argmax(dim=1)  # Get the predicted class by taking the index of the maximum probability

            # Append the true labels and predicted labels to their respective lists
            true_labels.extend(data.y.tolist())  # Convert true labels to a list and extend the list
            predicted_labels.extend(pred1.tolist())  # Convert predicted labels to a list and extend the list
            predicted_probs.extend(out.tolist())  # Convert predicted probabilities to a list and extend the list
    
    # Convert the lists to numpy arrays for metric calculations
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    predicted_probs = np.array(predicted_probs)

    # Calculate confusion matrix components (True Negative, False Positive, False Negative, True Positive)
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).flatten()
    print(tn, fp, fn, tp)  # Print the confusion matrix components
    
    # Calculate specificity (SP) and sensitivity (SE)
    sp = tn / (tn + fp)  # Specificity: TN / (TN + FP)
    se = tp / (tp + fn)  # Sensitivity: TP / (TP + FN)

    # Calculate accuracy (ACC) and area under the ROC curve (AUC)
    acc = accuracy_score(true_labels, predicted_labels)  # Calculate accuracy using sklearn's accuracy_score
    auc = roc_auc_score(true_labels, predicted_probs[:, 1])  # Calculate AUC score using sklearn's roc_auc_score (assumes binary classification)

    # Return the calculated metrics: accuracy, specificity, sensitivity, and AUC
    return acc, sp, se, auc


# If this script is being run directly (not imported as a module)
if __name__ == '__main__':  
    # Define the root directory paths for the training and test datasets
    train_root = '/MM-IDTarget/data/train'
    test_root = '/MM-IDTarget/data/test'
    
    # Instantiate the training and testing datasets by passing the root directories, raw data, and processed data file paths
    train_dataset = SMILES_Protein_Dataset(root=train_root, raw_dataset='train_data90_clusters.csv', processed_data='train_data90_clusters.pt')
    test_dataset = SMILES_Protein_Dataset(root=test_root, raw_dataset='test_data10_clusters.csv', processed_data='test_data10_clusters.pt')

    # Set the device to CUDA if a GPU is available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define the number of epochs for training
    epochs = 200
    
    # Initialize the model (MM_IDTarget) with specified parameters (input, output sizes, and embedding size) and move it to the selected device
    model = MM_IDTarget(3, 25 + 1, embedding_size=20).to(device)
    
    # Define the learning rate and weight decay for the optimizer
    lr = 0.0001
    weight_decay = 0.0005
    
    # Set the batch size for training and testing
    batch_size = 64
    
    # Set an initial value for the minimum loss to track and save the best model
    min = 10
    
    # Define the Adam optimizer with the specified learning rate and weight decay, applied to the model's parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Loop over the number of epochs for training
    for epoch in range(epochs):
        # Print the current epoch number
        print("Epoch: " + str(epoch + 1))
        
        # Initialize data loaders for training and testing with the specified batch size
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=False)
        test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, drop_last=False)
        
        # Open a file in append mode to log the results
        with open(f"/home/user/sgp/Chemgenomic/target/MM-IDTarget/result.txt", 'a') as f:
            
            # Initialize a progress bar for the training loop, showing the loss during training
            loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), colour='red', desc='Train Loss')
            
            # Call the train function to perform a training step and get the loss for this epoch
            loss = train()
            
            # Initialize a progress bar for the test loop, showing testing performance
            loop3 = tqdm(enumerate(test_dataloader), total=len(test_dataloader), colour='red', desc='Test')
            
            # Call the test function to evaluate the model on the test dataset and get the performance metrics
            acc3, sp3, se3, auc3 = test(loop3)
            
            # Print the training loss and evaluation metrics (accuracy, specificity, sensitivity, AUC) to the console
            print(loss)
            print(acc3, se3, sp3, auc3)
                
            # Write the current epoch's training loss to the result file
            f.write('Epoch {:03d}, Loss: {:.4f}\n'.format(epoch + 1, loss))
            
            # Write the test metrics (accuracy, specificity, sensitivity, AUC) to the result file
            f.write('Test:  Acc: {:.4f}, SP: {:.4f}, SE: {:.4f}, AUC: {:.4f}\n'
                    .format(acc3, sp3, se3, auc3))
            
            # If the current loss is lower than the previous minimum loss, save the model's state (weights)
            if loss < min:
                min = loss  # Update the minimum loss to the current loss
                # Save the model's state_dict (parameters) to a file
                torch.save(model.state_dict(), os.path.join(f"/home/user/sgp/Chemgenomic/target/MM-IDTarget/model.pkl"))
                   
        # Close the file after logging the results for the current epoch
        f.close()

