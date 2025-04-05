# Import necessary libraries
from torch_geometric.data import InMemoryDataset, Data  # Import InMemoryDataset and Data classes from torch_geometric
import torch  # Import PyTorch
import pandas as pd  # Import Pandas for handling CSV files
from rdkit import Chem  # Import RDKit for chemical molecule processing
import os  # Import os for file system operations
import numpy as np  # Import NumPy for array operations
from rdkit.Chem import AllChem  # Import AllChem for molecular fingerprint calculation
from src.DrugGraph import smiles_to_graph  # Import custom function smiles_to_graph to convert SMILES to graph structure

# Define protein vocabulary (mapping each amino acid to an integer)
VOCAB_PROTEIN = {"A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6,
                 "H": 7, "I": 8, "K": 9, "L": 10, "M": 11, "N": 12,
                 "P": 13, "Q": 14, "R": 15, "S": 16, "T": 17, "V": 18,
                 "W": 19, "Y": 20, "X": 21}

# Define reverse mapping for VOCAB_PROTEIN
NUM_TO_LETTER = {v:k for k, v in VOCAB_PROTEIN.items()}

# Convert amino acid sequence to integer representation
def seqs2int(target):
    return [VOCAB_PROTEIN[s] for s in target]

# Define SMILES symbols vocabulary (map each symbol to an integer)
VOCAB_SMILES = {
    'C': 1, 'c': 2, 'N': 3, 'n': 4, 'O': 5, 'o': 6, 'H': 7, 'h': 8, 'P': 9, 'p': 10, 'S': 11, 's': 12, 'F': 13, 'f': 14,
    'Cl': 15, 'Br': 16, 'I': 17, 'B': 18, 'b': 19, 'K': 20, 'k': 21, 'V': 22, 'v': 23, 'Mg': 24, 'm': 25, 'Li': 26, 'l': 27,
    'Zn': 28, 'z': 29, 'Si': 30, 'i': 31, 'As': 32, 'a': 33, 'Fe': 34, 'e': 35, 'Cu': 36, 'u': 37, 'Se': 38, 'se': 39,
    'R': 40, 'r': 41, 'T': 42, 't': 43, 'Na': 44, 'a': 45,'G': 46, 'g': 47, 'D': 48, 'd': 49, 'Q': 50, 'q': 51, 'X': 52, 
    'x': 53,'*': 54, '#': 55, '1': 56, '2': 57, '3': 58, '4': 59, '5': 60, '6': 61, '7': 62, '8': 63, '9': 64, '0': 65,
    '(': 66, ')': 67, '[': 68, ']': 69, '=': 70, '/': 71, '\\': 72, '@': 73, '+': 74, '-': 75, '.':76}

# Convert SMILES string to integer sequence
def smi2int(smile):
    return [VOCAB_SMILES[s] for s in smile]

# Function to calculate molecular fingerprints
def calculate_fingerprints(smiles):
    mol = Chem.MolFromSmiles(smiles)  # Create a molecule object from the SMILES string
    if mol is not None:  # If molecule object is valid
        ecfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)  # Compute ECFP4 fingerprint (radius 2, 1024 bits)
        ecfp4_array = [int(x) for x in ecfp4.ToBitString()]  # Convert fingerprint to a list of 0s and 1s
        return ecfp4_array  # Return the fingerprint array
    else:
        return None  # If molecule is invalid, return None

# Define custom dataset class inheriting from InMemoryDataset
class SMILES_Protein_Dataset(InMemoryDataset):
    # Initialization method, defining root directory, raw dataset path, processed data path, etc.
    def __init__(self, root, raw_dataset=None, processed_data=None, transform = None, pre_transform = None):
        self.root=root  # Dataset root directory
        self.raw_dataset=raw_dataset  # Raw dataset file
        self.processed_data=processed_data  # Processed data file
        self.max_smiles_len=256  # Maximum SMILES string length
        self.smiles_dict_len=65  # Length of the SMILES symbol dictionary
        super(SMILES_Protein_Dataset,self).__init__(root, transform, pre_transform)  # Call parent constructor
        if os.path.exists(self.processed_paths[0]):  # If processed file exists
            # self.process()  # Optional: Uncomment if you want to process again
            self.data, self.slices = torch.load(self.processed_paths[0])  # Load the processed data
        else:
            self.process()  # Process the data
            self.data, self.slices = torch.load(self.processed_paths[0])  # Load the processed data
    
    # Return the raw file names (list)
    @property
    def raw_file_names(self):
        return [self.raw_dataset]
    
    # Return the processed file names (list)
    @property
    def processed_file_names(self):
        return [self.processed_data]
    
    # Download dataset method (not needed here)
    def download(self):
        pass
    
    # Create processed data directory if it doesn't exist
    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)  # Create directory if it does not exist
        
    # Data processing function
    def process(self):
        # Read the CSV file
        data = pd.read_csv(self.raw_paths[0])  # Read the raw dataset
        data_list = []  # Create an empty list to store processed data
        for index, row in data.iterrows():  # Iterate over each row in the data
            # Extract filename part
            file_name = self.raw_paths[0].split("/")[-1]
            print(file_name[:-4] + " : " + str(index+1))  # Print the current file name and row number
            # Extract data from each row
            target_id = row['target_id']  # Extract target_id
            smile = row['smiles1']  # Extract SMILES string
            sequence = row['sequence']  # Extract sequence
            label = int(row['label'])  # Extract label and convert to integer
            drug_id = row['drug_id']  # Extract drug_id
            
            # Process Target part (protein data)
            df_proA = pd.read_csv("/MM-IDTarget/data/ProA.csv")  # Read protein-related data
            row_data_A = df_proA[df_proA['target_id'] == target_id]  # Find the row matching target_id
            row_data_list_A = row_data_A.values.tolist()[0][1:]  # Extract the relevant columns and convert to list
            pro_A = torch.tensor(row_data_list_A).unsqueeze(0)  # Convert to PyTorch tensor and add extra dimension
            
            # Process target sequence data
            target = seqs2int(sequence)  # Convert sequence to integer representation
            target_len = 1200  # Define fixed length for target sequence
            if len(target) < target_len:
                target = np.pad(target, (0, target_len - len(target)), mode='constant')  # Pad the sequence if it's shorter
            else:
                target = target[:target_len]  # Crop the sequence if it's longer
            target = torch.LongTensor([target])  # Convert to PyTorch tensor
            
            # Process target protein graph data (load preprocessed graph data)
            pro_graph = torch.load(f'/MM-IDTarget/data/protein_EW-GCN/{target_id}.pt')
            
            # Process drug data (calculate molecular fingerprints)
            ecfp4_array = calculate_fingerprints(smile)  # Calculate ECFP4 fingerprints
            ecfp4 = torch.tensor(ecfp4_array).unsqueeze(0)  # Convert fingerprints to PyTorch tensor and add extra dimension
            
            # Process SMILES graph structure
            smiles_graph = smiles_to_graph(smile)  # Convert SMILES string to graph structure
            
            # Process SMILES sequence
            smiles = smi2int(smile)  # Convert SMILES string to integer sequence
            if len(smiles) < 174:
                smiles = np.pad(smiles, (0, 174 - len(smiles)), mode='constant')  # Pad SMILES sequence to fixed length
            smiles = torch.LongTensor([smiles])  # Convert to PyTorch tensor
            
            # Create Data object to store feature data
            data = Data(smiles_graph = smiles_graph, pro_graph = pro_graph, pro_A = pro_A, target = target, 
                        smiles = smiles, ecfp4 = ecfp4, smile = smile, sequence = sequence, 
                        target_id = target_id, drug_id = drug_id, y = label)  # Create Data object
            data_list.append(data)  # Add Data object to list
        
        # Combine data list and save as processed data
        data, slices = self.collate(data_list)  # Use collate method to combine data
        torch.save((data, slices), self.processed_paths[0])  # Save the combined data

# Main program entry
if __name__ == '__main__':
    # Define the paths for training and testing sets
    train_root = '/MM-IDTarget/data/train'
    test_root = '/MM-IDTarget/data/test'
    
    # Create instances for training and testing datasets
    train_dataset = SMILES_Protein_Dataset(root=train_root, raw_dataset='train_data90_clusters.csv', processed_data='train_data90_clusters.pt')
    test_dataset = SMILES_Protein_Dataset(root=test_root, raw_dataset='test_data10_clusters.csv', processed_data='test_data10_clusters.pt')
