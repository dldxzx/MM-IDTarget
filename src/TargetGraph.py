# Importing necessary libraries
import dgl  # Import Deep Graph Library for graph-based machine learning
import math  # Import math module for mathematical functions
import torch as th  # Import PyTorch for deep learning tasks, aliasing as 'th'
import numpy as np  # Import NumPy for numerical operations
from Bio.PDB import PDBParser  # Import PDBParser from Biopython for parsing protein structure files
import numpy as np  # Import NumPy again (duplicate, can be removed)
import pandas as pd  # Import Pandas for data manipulation and analysis
import torch  # Import PyTorch for tensor computations
import warnings  # Import warnings module to manage warnings
warnings.filterwarnings("ignore")  # Ignore warnings

# Normalize a dictionary of values to a 0-1 scale
def dic_normalize(dic):
    # Get the maximum and minimum values from the dictionary
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # Calculate the range (max - min)
    interval = float(max_value) - float(min_value)
    # Normalize each value in the dictionary
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    # Add the midpoint value (for reference)
    dic['X'] = (max_value + min_value) / 2.0
    return dic

# Dictionary of amino acid three-letter to one-letter code
res_dict = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# Lists of amino acids categorized by their properties
pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

# Amino acid residue weight table (in daltons)
res_weight_table = {
    'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14, 'I': 113.16, 'K': 128.18,
    'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13, 'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13,
    'W': 186.22, 'Y': 163.18
}

# pKa, pKb, pkx, and other tables for residues
res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36, 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21, 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}
res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17, 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13, 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}
res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00, 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00, 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}
res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59, 'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65, 'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}
res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100, 'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7, 'T': 13, 'V': 79, 'W': 84, 'Y': 49}
res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99, 'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5, 'T': 13, 'V': 76, 'W': 97, 'Y': 63}

# Normalizing all the residue property tables
res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)

# Function to perform one-hot encoding for a residue given an allowable set of values
def one_of_k_encoding(x, allowable_set):
    # If the residue is not in the allowable set, raise an exception
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    # Return a list where the position of 'x' is marked with a 1, others are 0
    return list(map(lambda s: x == s, allowable_set))

# Function to perform one-hot encoding for a residue, maps unknown residues to the last element of allowable set
def one_of_k_encoding_unk(x, allowable_set):
    # If residue is not in the allowable set, map it to the last element
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

# Function to extract various features from a given residue
def residue_features(residue):
    # Boolean properties for aliphatic, aromatic, neutral, acidic, and basic charged residues
    res_property1 = [
        1 if residue in pro_res_aliphatic_table else 0, 
        1 if residue in pro_res_aromatic_table else 0,
        1 if residue in pro_res_polar_neutral_table else 0,
        1 if residue in pro_res_acidic_charged_table else 0,
        1 if residue in pro_res_basic_charged_table else 0
    ]
    # Numeric properties for weight, pKa, pKb, pkx, and hydrophobicity
    res_property2 = [
        res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], 
        res_pkx_table[residue], res_pl_table[residue], res_hydrophobic_ph2_table[residue], 
        res_hydrophobic_ph7_table[residue]
    ]
    return np.array(res_property1 + res_property2)  # Combine all properties into a single array

# Function to convert a sequence of residues into a feature matrix
def seq_feature(pro_seq):
    # Initialize feature matrices
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))  # One-hot encoding for residues
    pro_property = np.zeros((len(pro_seq), 12))  # Properties matrix
    for i in range(len(pro_seq)):
        # Perform one-hot encoding and feature extraction for each residue
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    # Concatenate the one-hot encoding and property matrices
    return np.concatenate((pro_hot, pro_property), axis=1)

# Function to compute the cosine similarity between two vectors
def cos_sim(vec1, vec2):
    # Convert vectors to numpy arrays
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    # Calculate cosine similarity
    cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cos_sim  # Return cosine similarity value (range [-1, 1])

# Function to calculate the angle between three points (vectors) in 3D space
def cal_angle(point_a, point_b, point_c):
    # Extract coordinates of points
    a_x, b_x, c_x = point_a[0], point_b[0], point_c[0]  # x-coordinates
    a_y, b_y, c_y = point_a[1], point_b[1], point_c[1]  # y-coordinates
    if len(point_a) == len(point_b) == len(point_c) == 3:
        a_z, b_z, c_z = point_a[2], point_b[2], point_c[2]  # z-coordinates for 3D points
    else:
        a_z, b_z, c_z = 0, 0, 0  # Default z=0 for 2D points

    # Calculate vectors from point_b to point_a and point_b to point_c
    x1, y1, z1 = (a_x - b_x), (a_y - b_y), (a_z - b_z)
    x2, y2, z2 = (c_x - b_x), (c_y - b_y), (c_z - b_z)

    # Compute the cosine of the angle between the two vectors
    cos_b = (x1 * x2 + y1 * y2 + z1 * z2) / (math.sqrt(x1**2 + y1**2 + z1**2) * math.sqrt(x2**2 + y2**2 + z2**2))
    B = math.degrees(math.acos(cos_b))  # Convert cosine value to angle in degrees
    return cos_b  # Return cosine value of the angle (range [-1, 1])


# Function to convert contact matrix and distance matrix into a graph representation
def TargetToGraph(contact_matrix, distance_matrix, ca_coords, seq3, sequence, contact=1, dis_min=1):
    c_size = len(contact_matrix)  # Get the size of the contact matrix (number of residues)
    
    # Initialize an empty DGL graph
    G = dgl.DGLGraph()
    G.add_nodes(c_size)  # Add nodes corresponding to residues (amino acids)

    # Convert 3-letter sequence (seq3) into the full amino acid sequence
    seq = ''
    for i in range(0, len(seq3), 3):
        triplet = seq3[i:i+3]  # Extract triplet of letters
        converted_triplet = res_dict[triplet]  # Convert triplet to 1-letter amino acid code
        seq += converted_triplet  # Append to sequence string
    
    # Generate node features based on the sequence (e.g., using one-hot encoding or other methods)
    node_features = seq_feature(seq)  # seq_feature should generate features for each residue

    # List to store edge features
    edge_features = []
    
    # Loop over the contact matrix to identify contacts and calculate edge features
    for i in range(len(contact_matrix)):
        for j in range(len(contact_matrix)):
            contact_ij = contact_matrix[i][j]
            
            if i != j and contact_ij == contact:  # Check if i and j are different residues and are in contact
                G.add_edges(i, j)  # Add an edge between i and j
                
                # Calculate cosine similarity between node features (amino acid properties)
                sim_ij = cos_sim(node_features[i], node_features[j])[0, 0]  # Cosine similarity score
                
                # Calculate distance feature (either 1 or scaled by inverse of distance)
                if distance_matrix[i][j] <= dis_min:
                    dis_ij = dis_min  # Use dis_min if distance is below threshold
                else:
                    dis_ij = 1 / distance_matrix[i][j]  # Use inverse distance
                
                # Calculate angle between residues based on C-alpha coordinates
                angle_ij = cal_angle(ca_coords[i], [0, 0, 0], ca_coords[j])  # Angle between the two residues' C-alpha atoms
                
                # Create edge feature vector consisting of similarity, distance, and angle
                contact_features_ij = [sim_ij, dis_ij, angle_ij]
                print(sim_ij, dis_ij, angle_ij)  # Print the calculated features for debugging
                
                # Append the edge feature vector to the list
                edge_features.append(contact_features_ij)

    # Set node features in the graph
    G.ndata['x'] = th.from_numpy(np.array(node_features)).to(th.float32)
    
    # Set edge features in the graph
    G.edata['w'] = th.from_numpy(np.array(edge_features)).to(th.float32)
    
    # Return the graph object
    return G

# Function to calculate contact and distance matrices from a protein structure file
def calculate_contact_distance_matrices(structure_file, seq):
    parser = PDBParser()  # Initialize PDB parser from Biopython
    structure = parser.get_structure('protein', structure_file)  # Parse the structure file

    # Initialize sequence and C-alpha coordinates lists
    sequence = ""
    ca_coords = []

    # Iterate over the structure to extract amino acid sequence and C-alpha atom coordinates
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == ' ' and residue.get_resname() != "HOH":  # Exclude water molecules (HOH)
                    sequence += residue.get_resname()  # Append amino acid name to sequence
                    if residue.has_id('CA'):  # If C-alpha atom exists
                        ca_atom = residue['CA']  # Get the C-alpha atom
                        ca_coords.append(ca_atom.get_coord())  # Append its coordinates to the list

    # Calculate contact and distance matrices based on C-alpha coordinates
    num_residues = len(ca_coords)  # Get number of residues
    contact_matrix = np.zeros((num_residues, num_residues), dtype=bool)  # Initialize contact matrix
    distance_matrix = np.zeros((num_residues, num_residues))  # Initialize distance matrix

    # Loop over residue pairs and calculate distances
    for i in range(num_residues):
        for j in range(i+1, num_residues):
            distance = np.linalg.norm(ca_coords[i] - ca_coords[j])  # Calculate Euclidean distance between C-alpha atoms
            if distance <= 8.0:  # Define contact threshold (8 Å)
                contact_matrix[i, j] = True
                contact_matrix[j, i] = True
            distance_matrix[i, j] = distance  # Store the distance
            distance_matrix[j, i] = distance  # Distance matrix is symmetric

    # Call TargetToGraph to generate graph from contact and distance matrices
    return TargetToGraph(contact_matrix, distance_matrix, ca_coords, sequence, seq, contact=1, dis_min=1)

# Read target data from CSV file
data = pd.read_csv('/home/user/sgp/Chemgenomic/target/MM-IDTarget/data/target.csv')

# Flag to indicate whether to start saving the .pt files
start_saving = False

# Iterate through rows in the data frame
for index, row in data.iterrows():
    target_id = row['target_id']
    
    # Start saving when a target_id of 'T1' is encountered
    if target_id == 'T1':
        start_saving = True
    
    if start_saving:
        sequence = row['sequence']  # Get the protein sequence from the row
        structure_file = f'/home/user/sgp/Chemgenomic/target/MM-IDTarget/data/structure/{target_id}.pdb'  # Path to PDB file
        
        # Generate graph from the contact and distance matrices
        G = calculate_contact_distance_matrices(structure_file, sequence)
        print(G)  # Print the graph for debugging
        
        # Save the generated graph to a .pt file
        save_path = f'/home/user/sgp/Chemgenomic/target/MM-IDTarget/data/protein_EW-GCN/{target_id}.pt'
        torch.save(G, save_path)  # Save graph to the specified path

    # Optionally exit the loop after saving a target, if needed
    # exit()

