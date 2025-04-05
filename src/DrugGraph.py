# Import necessary libraries
import dgl  # Graph neural network library for processing graph-structured data
import numpy as np  # Numerical computation library for array and matrix operations
import torch  # Deep learning library for constructing and training neural networks
from rdkit import Chem  # For chemical molecule processing and SMILES parsing
from scipy import sparse as sp  # Sparse matrix library
import os  # For file and environment variable operations
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Solve some environment variable issues to avoid conflicts
import warnings  # For controlling warning messages
warnings.filterwarnings("ignore")  # Ignore warning messages

# Define a dictionary to map characters in SMILES strings to numeric indices
CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

# Convert SMILES string to a numeric label array, maximum length of MAX_SMI_LEN
def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64())  # Create an array of zeros with length MAX_SMI_LEN
    for i, ch in enumerate(line[:MAX_SMI_LEN]):  # Iterate over the SMILES string and convert characters to indices
        X[i] = smi_ch_ind[ch]  # Convert character to index based on the mapping
    return X  # Return the numeric label array

# One-hot encoding: map x to a value in the allowable set
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:  # Check if x is in the allowable set
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return [x == s for s in allowable_set]  # Return a boolean list with True where x matches a value in the set

# One-hot encoding: map input not in the allowable set to the last element
def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:  # If x is not in the allowable set, map it to the last element
        x = allowable_set[-1]
    return [x == s for s in allowable_set]  # Return a boolean list indicating if x matches any element in the set

# Calculate the Laplacian positional encoding of the graph
def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding via Laplacian eigenvectors
    """
    # Calculate the Laplacian matrix
    A = g.adjacency_matrix(scipy_fmt='csr').astype(float)  # Get the adjacency matrix
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)  # Calculate the degree matrix's square root
    L = sp.eye(g.number_of_nodes()) - N * A * N  # Compute the Laplacian matrix

    # Compute eigenvalues and eigenvectors of the Laplacian matrix
    EigVal, EigVec = np.linalg.eig(L.toarray())  # Compute eigenvalues and eigenvectors
    idx = EigVal.argsort()  # Sort the eigenvalues in ascending order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])  # Reorder the eigenvectors according to the sorted eigenvalues
    if EigVec.shape[1] < pos_enc_dim + 1:  # If the number of eigenvectors is less than pos_enc_dim, pad them
        PadVec = np.zeros((EigVec.shape[0], pos_enc_dim + 1 - EigVec.shape[1]), dtype=EigVec.dtype)
        EigVec = np.concatenate((EigVec, PadVec), 1)  # Pad the eigenvector matrix
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()  # Assign the eigenvectors to graph node features
    return g  # Return the graph with positional encoding added

# Generate atom features, considering atom symbol, degree, formal charge, etc.
def atom_features(atom, explicit_H=False, use_chirality=True):
    """Generate atom features including atom symbol(17), degree(7), formal charge(1),
    radical electrons(1), hybridization(6), aromatic(1), hydrogen atoms attached(5), Chirality(3)
    """
    # Define possible atom symbols, degrees, and hybridization types
    symbol = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'other']  # 17 dimensions
    degree = [0, 1, 2, 3, 4, 5, 6]  # 7 dimensions
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2,
                         'other']  # 6 dimensions
    # Concatenate all features
    results = one_of_k_encoding_unk(atom.GetSymbol(), symbol) + \
              one_of_k_encoding(atom.GetDegree(), degree) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [
                  atom.GetIsAromatic()]  # 17+7+2+6+1=33

    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])  # 33+5=38
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 38+3 =41
    return results  # Return the atom features

# Generate bond features, including bond type, conjugation, ring involvement, stereochemistry, etc.
def bond_features(bond, use_chirality=True):
    """Generate bond features including bond type(4), conjugated(1), in ring(1), stereo(4)"""
    bt = bond.GetBondType()  # Get bond type
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),  # Whether conjugated
        bond.IsInRing()  # Whether in a ring
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])  # Stereochemical type
    return np.array(bond_feats).astype(int)  # Return bond features as integer array

# Convert SMILES string to a graph structure
def smiles_to_graph(smiles, explicit_H=False, use_chirality=True):
    try:
        mol = Chem.MolFromSmiles(smiles)  # Generate molecule object from SMILES string
    except:
        raise RuntimeError("SMILES cannot be parsed!")  # Raise error if SMILES parsing fails
    g = dgl.DGLGraph()  # Create an empty graph object
    # Add nodes
    num_atoms = mol.GetNumAtoms()  # Get the number of atoms
    g.add_nodes(num_atoms)  # Add nodes to the graph

    # Get features for each atom
    atom_feats = np.array([atom_features(a, explicit_H=explicit_H) for a in mol.GetAtoms()])
    if use_chirality:
        chiralcenters = Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True,
                                                  useLegacyImplementation=False)  # Find chiral centers
        chiral_arr = np.zeros([num_atoms, 3])  # Create array for chiral features
        for (i, rs) in chiralcenters:
            if rs == 'R':
                chiral_arr[i, 0] = 1
            elif rs == 'S':
                chiral_arr[i, 1] = 1
            else:
                chiral_arr[i, 2] = 1
        atom_feats = np.concatenate([atom_feats, chiral_arr], axis=1)  # Add chiral information to atom features

    g.ndata["atom"] = torch.tensor(atom_feats)  # Assign atom features to graph node data

    # Add edges
    src_list = []
    dst_list = []
    bond_feats_all = []
    num_bonds = mol.GetNumBonds()  # Get number of bonds
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)  # Get bond by index
        u = bond.GetBeginAtomIdx()  # Get start atom index
        v = bond.GetEndAtomIdx()  # Get end atom index
        bond_feats = bond_features(bond, use_chirality=use_chirality)  # Get bond features
        src_list.extend([u, v])  # Add source atoms for the edge
        dst_list.extend([v, u])  # Add destination atoms for the edge
        bond_feats_all.append(bond_feats)  # Append bond features
        bond_feats_all.append(bond_feats)  # Add bidirectional edge, so append twice

    g.add_edges(src_list, dst_list)  # Add edges to the graph

    g.edata["bond"] = torch.tensor(np.array(bond_feats_all))  # Assign bond features to graph edge data
    g = laplacian_positional_encoding(g, pos_enc_dim=8)  # Add Laplacian positional encoding
    return g  # Return the graph object

    
# The following code is typically used to read data and convert SMILES to graphs
# df = pd.read_csv('/home/user/fpk/sgp/Chemgenomic/target/pre_data/drug.csv')
# smiles_column = df['Canonical_SMILES']
# index = 1
# for smiles in smiles_column:
#     print(str(index) + ":" + smiles)
#     index = index + 1
#     graph = smiles_to_graph(smiles)
#     print(graph)
