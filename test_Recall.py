# Import the torch library for deep learning and tensor operations
import torch

# Import pandas for data manipulation and handling data structures like DataFrame
import pandas as pd

# Import the Chem module from RDKit, a toolkit for cheminformatics
from rdkit import Chem

# Import os for interacting with the operating system, such as file paths
import os

# Import the MM_IDTarget class from the MM_IDTarget_Recall module, located in the src/models folder
from src.models.MM_IDTarget_Recall import MM_IDTarget

# Import the smiles_to_graph function from the src/DrugGraph module to convert SMILES strings to graph representations
from src.DrugGraph import smiles_to_graph

# Import AllChem from RDKit for advanced cheminformatics tools like molecular fingerprints
from rdkit.Chem import AllChem

# Import logging to log messages for tracking and debugging purposes
import logging

# Import numpy for numerical operations and working with arrays
import numpy as np

# Import utils from the DeepPurpose library, which contains utilities for drug-target interaction modeling
from DeepPurpose import utils

# Import the DTI (Drug-Target Interaction) model from DeepPurpose to build or load models for predicting drug-target interactions
from DeepPurpose import DTI as models

# Disable warnings to avoid cluttering the output with unnecessary warnings
import warnings
warnings.filterwarnings("ignore")
    

# Set device to CUDA if available, otherwise use CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Initialize the model (MM_IDTarget) with the given dimensions
Models = MM_IDTarget(3, 25 + 1, embedding_size=128).to(device)

# Define a dictionary to map amino acid characters to integers
VOCAB_PROTEIN = {"A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6,
                 "H": 7, "I": 8, "K": 9, "L": 10, "M": 11, "N": 12,
                 "P": 13, "Q": 14, "R": 15, "S": 16, "T": 17, "V": 18,
                 "W": 19, "Y": 20, "X": 21}

# Function to convert amino acid sequence to integer representation
def seqs2int(target):
    return [VOCAB_PROTEIN[s] for s in target]

# Define a dictionary to map chemical elements in SMILES notation to integers
VOCAB_SMILES = {
    'C': 1, 'c': 2, 'N': 3, 'n': 4, 'O': 5, 'o': 6, 'H': 7, 'h': 8, 'P': 9, 'p': 10, 'S': 11, 's': 12, 'F': 13, 'f': 14,
    'Cl': 15, 'Br': 16, 'I': 17, 'B': 18, 'b': 19, 'K': 20, 'k': 21, 'V': 22, 'v': 23, 'Mg': 24, 'm': 25, 'Li': 26, 'l': 27,
    'Zn': 28, 'z': 29, 'Si': 30, 'i': 31, 'As': 32, 'a': 33, 'Fe': 34, 'e': 35, 'Cu': 36, 'u': 37, 'Se': 38, 'se': 39,
    'R': 40, 'r': 41, 'T': 42, 't': 43, 'Na': 44, 'a': 45,'G': 46, 'g': 47, 'D': 48, 'd': 49, 'Q': 50, 'q': 51, 'X': 52, 
    'x': 53,'*': 54, '#': 55, '1': 56, '2': 57, '3': 58, '4': 59, '5': 60, '6': 61, '7': 62, '8': 63, '9': 64, '0': 65,
    '(': 66, ')': 67, '[': 68, ']': 69, '=': 70, '/': 71, '\\': 72, '@': 73, '+': 74, '-': 75, '.':76}

# Reverse dictionary to map integers back to SMILES characters
NUM_TO_LETTER = {v: k for k, v in VOCAB_SMILES.items()}

# Function to convert SMILES string into integer representation
def smi2int(smile):
    return [VOCAB_SMILES[s] for s in smile]

# Define the list of possible characters for SMILES strings
all_characters = ['C', 'c', 'N', 'n', 'O', 'o', 'H', 'h', 'P', 'p', 'S', 's', 'F', 'f', 'Cl', 'Br', 'I', 'B', 'b', 
                  'K', 'k', 'V', 'v', 'Mg', 'm', 'Li', 'l', 'Zn', 'z', 'Si', 'i', 'As', 'a', 'Fe', 'e', 'Cu', 'u', 
                  'Se', 'se', 'R', 'r', 'T', 't', 'Na', 'a', 'G', 'g', 'D', 'd', 'Q', 'q', 'X', 'x', '*', '#', '1', 
                  '2', '3', '4', '5', '6', '7', '8', '9', '0', '(', ')', '[', ']', '=', '/', '\\', '@', '+', '-', '.']

# Define a function to calculate molecular fingerprints (ECFP4)
def calculate_fingerprints(smiles):
    mol = Chem.MolFromSmiles(smiles)  # Convert SMILES to molecule object
    if mol is not None:
        # Compute the ECFP4 fingerprint with 1024 bits
        ecfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        # Convert the fingerprint to a list of bits (0 or 1)
        ecfp4_array = [int(x) for x in ecfp4.ToBitString()]
        return ecfp4_array
    else:
        return None

# Function to create a dataset containing SMILES and protein sequences for a given target
def SMILES_Protein_Dataset(sequence, smile, target_id):
    # Calculate the ECFP4 fingerprint for the SMILES string
    ecfp4_array = calculate_fingerprints(smile)
    ecfp4 = torch.tensor(ecfp4_array).unsqueeze(0)  # Convert fingerprint to tensor
    df_proA = pd.read_csv("/MM-IDTarget/data/ProA.csv")  # Load protein data
    row_data_A = df_proA[df_proA['target_id'] == target_id]  # Extract the row corresponding to the target_id
    row_data_list_A = row_data_A.values.tolist()[0][1:]  # Get protein data excluding the first column (target_id)
    pro_A = torch.tensor(row_data_list_A).unsqueeze(0)  # Convert to tensor
    target = seqs2int(sequence)  # Convert protein sequence to integer representation
    target_len = 1200  # Set target length
    if len(target) < target_len:
        target = np.pad(target, (0, target_len - len(target)), mode='constant')  # Pad sequence if necessary
    else:
        target = target[:target_len]  # Truncate sequence if too long
    target = torch.LongTensor([target])  # Convert to tensor
    smiles_graph = smiles_to_graph(smile)  # Convert SMILES to graph representation
    pro_graph = torch.load(f'/MM-IDTarget/data/protein_EW-GCN/{target_id}.pt')  # Load protein graph
    smiles = smi2int(smile)  # Convert SMILES to integer sequence
    if len(smiles) < 174:
        smiles = np.pad(smiles, (0, 174 - len(smiles)), mode='constant')  # Pad SMILES if necessary
    smiles = torch.LongTensor([smiles])  # Convert to tensor

    # Perform inference with the model without computing gradients
    with torch.no_grad():
        out = Models(smiles_graph.to(device), pro_graph.to(device), pro_A.to(device), target.to(device), smiles.to(device), ecfp4.to(device))
        second_column = out[:, 1]  # Extract the second column of the model's output
    return second_column

# Function to process sorted data based on the given criteria
def process_data_function(sorted_targets, matching_rows, drug_encoding, target_encoding, ff2_modelss, smiles, target):
    # Filter rows that match the target and have a higher score
    matching_rows = [(index, row) for index, row in enumerate(sorted_targets) if row[1] == target]
    filtered_rows = [row for row in sorted_targets if row[2] > matching_rows[0][1][2]]
    filtered_rows1 = [row for row in sorted_targets if row[2] == matching_rows[0][1][2]]
    
    # If there are more than 10 matching rows, print the result and calculate the index
    if len(filtered_rows) > 10:
        print(f"Result:")
        print("Result source: " + str(matching_rows[0][1][2]) + ", >= source number: " + str(len(filtered_rows)), str(len(filtered_rows1)))
        ff2_indexs = len(filtered_rows) + 1
    else:
        data_DTA = [row for row in sorted_targets if row[2] == matching_rows[0][1][2]]  # Filter DTA data with matching score
        ff2_DTA_list_all = []
        for data_dta in data_DTA:  
            # Process DTA data for prediction
            ff2_X_pred = utils.data_process([smiles], [data_dta[1]], [0], drug_encoding, target_encoding, split_method='no_split')
            ff2_DTA_sorce = ff2_modelss.predict(ff2_X_pred)  # Predict the DTA score
            ff2_DTA_list_all.append((data_dta[0], data_dta[1], ff2_DTA_sorce[0])) 
            if data_dta[1] == target:
                ff2_matching_rows2_all = ff2_DTA_sorce[0]
        ff2_sorted_targets_dta_all = sorted([x[2] for x in ff2_DTA_list_all], reverse=True)  # Sort the DTA scores
        ff2_filtered_rows21_all = [row for row in ff2_sorted_targets_dta_all if row > ff2_matching_rows2_all]  # Filter scores greater than the target's DTA score
        ff2_filtered_rows22_all = [row for row in ff2_sorted_targets_dta_all if row == ff2_matching_rows2_all]  # Filter matching DTA scores
        ff2_indexs = len(ff2_filtered_rows21_all) + len(filtered_rows)  # Calculate the index based on filtered rows
        print(f"Result:")
        print("Result source: " + str(matching_rows[0][1][2]) + ", >= source number: " + str(len(filtered_rows)), str(len(filtered_rows1)))
        print("DTA source all: " + str(ff2_matching_rows2_all) + ", top: " + str(ff2_indexs) + ", DTA >= source number: " + str(len(ff2_filtered_rows21_all)), str(len(ff2_filtered_rows22_all)))

    return ff2_indexs
    
            
# Define a function to recall top 859 predictions based on a pre-trained model
def Recall_Top859(pretrained_model, target, smiles, target_id):
    # Check if the pre-trained model path exists and is not an empty string
    if os.path.exists(pretrained_model) and pretrained_model != " ":
        logging.info(f"Starting to load the MMGAT model")
        try:
            # Load the state dictionary of the pre-trained model
            state_dict = torch.load(pretrained_model)
            new_model_state_dict = Models.state_dict()
            
            # Try to update the model's state dictionary with the pre-trained model's state dictionary
            for key in new_model_state_dict.keys():
                if key in state_dict.keys():
                    try:
                        # Copy the parameters from the pre-trained model
                        new_model_state_dict[key].copy_(state_dict[key])
                    except:
                        None
            # Load the updated state dictionary into the model
            Models.load_state_dict(new_model_state_dict)
            logging.info("MMGAT model loaded successfully (!)")
        except:
            # If an error occurs, adjust the state dictionary keys and reload the model
            new_state_dict = {}
            for key, value in state_dict.items():
                new_state_dict[key.replace("module.", "")] = value
            Models.load_state_dict(new_state_dict)
            logging.info("MMGAT model loaded successfully (!!!)")
    else:
        # Log if the model path does not exist or is empty
        logging.info("Model path does not exist, unable to load the model")
    
    # Set the model to evaluation mode
    Models.eval()
    
    # Read the target data from a CSV file
    df1 = pd.read_csv('/MM-IDTarget/data/target.csv')  
    try:
        # Attempt to retrieve the target_id corresponding to the provided target sequence
        target_id = df1.loc[df1['sequence'] == target]['target_id'].values[0]
    except IndexError:
        # If an IndexError occurs (i.e., the target is not found), log the error and skip the iteration
        print("error")
    
    # Initialize an empty list to store the target data
    target_list = []
    
    # Loop over a range to process multiple target sequences (from 1 to 859)
    for j in range(1, 860):
        # Read the target data from the CSV file again
        df1 = pd.read_csv('/MM-IDTarget/data/target.csv')
        
        # Dynamically create a target_id based on the loop index
        target_id = f"T{j}"
        
        # Retrieve the row of the dataframe where the target_id matches the current target_id
        row1 = df1.loc[df1['target_id'] == target_id]
        
        # Extract the sequence for the current target_id
        sequence = row1['sequence'].values[0]
        
        # Create a dataset object using the sequence, smiles, and target_id
        data = SMILES_Protein_Dataset(sequence, smiles, target_id)
        
        # Append the target_id, sequence, and dataset to the target_list
        target_list.append((target_id, sequence, data.item()))
    
    # Sort the target list based on the third element (data) in descending order
    sorted_targets = sorted(target_list, key=lambda x: x[2], reverse=True)
    
    # Find the matching rows in the sorted list where the sequence matches the target sequence
    matching_rows = [(index, row) for index, row in enumerate(sorted_targets) if row[1] == target]
    
    # If there are matching rows, filter the sorted targets list to include only those with scores above the matched sequence's score
    if len(matching_rows) != 0:
        # Create a filtered list with rounded values for readability
        filtered_sorted_targets = [
            (row[0], row[1], round(row[2], 2)) for row in sorted_targets[:next((index for index, row in enumerate(sorted_targets) if row[2] < matching_rows[0][1][2]), len(sorted_targets))]
        ]
        
        # Define the encodings for drugs and targets
        drug_encoding, target_encoding = 'MPNN', 'CNN'
        
        # Load a pre-trained model for prediction
        modelss = models.model_pretrained(path_dir=f'/MM-IDTarget/reslut_model/model_MPNN_CNN') 
        
        # Process the filtered data and return the corresponding indices
        indexs = process_data_function(filtered_sorted_targets, matching_rows, drug_encoding, target_encoding, modelss, smiles, target)
        
    # Return the index found, incremented by 1
    return indexs + 1

# Define the target sequence, SMILES representation, and target_id to pass into the function
target = 'MARRCGPVALLLGFGLLRLCSGVWGTDTEERLVEHLLDPSRYNKLIRPATNGSELVTVQLMVSLAQLISVHEREQIMTTNVWLTQEWEDYRLTWKPEEFDNMKKVRLPSKHIWLPDVVLYNNADGMYEVSFYSNAVVSYDGSIFWLPPAIYKSACKIEVKHFPFDQQNCTMKFRSWTYDRTEIDLVLKSEVASLDDFTPSGEWDIVALPGRRNENPDDSTYVDITYDFIIRRKPLFYTINLIIPCVLITSLAILVFYLPSDCGEKMTLCISVLLALTVFLLLISKIVPPTSLDVPLVGKYLMFTMVLVTFSIVTSVCVLNVHHRSPTTHTMAPWVKVVFLEKLPALLFMQQPRHHCARQRLRLRRRQREREGAGALFFREAPGADSCTCFVNRASVQGLAGAFGAEPAPVAGPGRSGEPCGCGLREAVDGVRFIADHMRSEDDDQSVSEDWKYVAMVIDRLFLWIFVFVCVFGTIGMFLQPLFQNYTTTTFLHSDHSAPSSK'
smiles = 'N1C[C@@H]2C[C@H](C1)c1c2cc2c(c1)nccn2'
target_id = 'T633'

# Call the Recall_Top859 function with the given parameters and print the result
index = Recall_Top859(f"/MM-IDTarget/reslut_model/MM-IDTarget_model.pkl", target, smiles, target_id)
print(index)
