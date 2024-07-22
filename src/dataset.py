from torch_geometric.data import InMemoryDataset, Data
import torch
import pandas as pd
from rdkit import Chem
import os
import numpy as np
from rdkit.Chem import AllChem
from src.DrugGraph import smiles_to_graph
    

VOCAB_PROTEIN = {"A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6,
                 "H": 7, "I": 8, "K": 9, "L": 10, "M": 11, "N": 12,
                 "P": 13, "Q": 14, "R": 15, "S": 16, "T": 17, "V": 18,
                 "W": 19, "Y": 20, "X": 21}

NUM_TO_LETTER = {v:k for k, v in VOCAB_PROTEIN.items()}

def seqs2int(target):
    return [VOCAB_PROTEIN[s] for s in target]

VOCAB_SMILES = {
    'C': 1, 'c': 2, 'N': 3, 'n': 4, 'O': 5, 'o': 6, 'H': 7, 'h': 8, 'P': 9, 'p': 10, 'S': 11, 's': 12, 'F': 13, 'f': 14,
    'Cl': 15, 'Br': 16, 'I': 17, 'B': 18, 'b': 19, 'K': 20, 'k': 21, 'V': 22, 'v': 23, 'Mg': 24, 'm': 25, 'Li': 26, 'l': 27,
    'Zn': 28, 'z': 29, 'Si': 30, 'i': 31, 'As': 32, 'a': 33, 'Fe': 34, 'e': 35, 'Cu': 36, 'u': 37, 'Se': 38, 'se': 39,
    'R': 40, 'r': 41, 'T': 42, 't': 43, 'Na': 44, 'a': 45,'G': 46, 'g': 47, 'D': 48, 'd': 49, 'Q': 50, 'q': 51, 'X': 52, 
    'x': 53,'*': 54, '#': 55, '1': 56, '2': 57, '3': 58, '4': 59, '5': 60, '6': 61, '7': 62, '8': 63, '9': 64, '0': 65,
    '(': 66, ')': 67, '[': 68, ']': 69, '=': 70, '/': 71, '\\': 72, '@': 73, '+': 74, '-': 75, '.':76}

def smi2int(smile):
    # 将 SMILES 字符串转换成整数序列
    return [VOCAB_SMILES[s] for s in smile]

# 定义函数来计算分子指纹
def calculate_fingerprints(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        ecfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)  # 计算ECFP4指纹
        ecfp4_array = [int(x) for x in ecfp4.ToBitString()]  # 将ECFP4指纹转换为0和1组成的数组
        return ecfp4_array
    else:
        return None, None
    

class SMILES_Protein_Dataset(InMemoryDataset):
    def __init__(self, root, raw_dataset=None, processed_data=None, transform = None, pre_transform = None):
        self.root=root
        self.raw_dataset=raw_dataset
        self.processed_data=processed_data
        self.max_smiles_len=256
        self.smiles_dict_len=65
        super(SMILES_Protein_Dataset,self).__init__(root, transform, pre_transform)
        if os.path.exists(self.processed_paths[0]):
            # self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])
        
    # 原始文件位置
    @property
    def raw_file_names(self):
        return [self.raw_dataset]
    
    # 文件保存位置
    @property
    def processed_file_names(self):
        return [self.processed_data]
        # return []
    
    def download(self):
        pass
    
    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)   
        
    def process(self):
        # 读取CSV文件
        data = pd.read_csv(self.raw_paths[0])
        data_list = []
        for index, row in data.iterrows():
            # 使用字符串操作获取文件名部分
            file_name = self.raw_paths[0].split("/")[-1]
            print(file_name[:-4] + " : " + str(index+1))
            # 获取每一行数据
            target_id = row['target_id']
            smile = row['smiles1']
            sequence = row['sequence']
            label = int(row['label'])
            drug_id = row['drug_id']
            
            # Target
            # proA 物理化学性质
            df_proA = pd.read_csv("/MM-IDTarget/data/ProA.csv")
            row_data_A = df_proA[df_proA['target_id'] == target_id] # 获取 target_id 等于 "T1" 的那一行数据
            row_data_list_A = row_data_A.values.tolist()[0][1:] # 提取所需的列数据并转换为列表
            pro_A = torch.tensor(row_data_list_A).unsqueeze(0) # 将列表转换为 PyTorch 张量
            # Target CNN
            target = seqs2int(sequence)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len - len(target)),mode='constant')
            else:
                target = target[:target_len]
            target=torch.LongTensor([target])
            #EW-GCN
            pro_graph = torch.load(f'/MM-IDTarget/data/protein_EW-GCN/{target_id}.pt')
            
            # Drug
            # ecpf4
            ecfp4_array = calculate_fingerprints(smile)
            ecfp4 = torch.tensor(ecfp4_array).unsqueeze(0)
            # GraphTransformer
            smiles_graph = smiles_to_graph(smile)
            # SMILES CNN
            smiles = smi2int(smile)
            if len(smiles) < 174:
                smiles = np.pad(smiles, (0, 174 - len(smiles)),mode='constant')
            smiles = torch.LongTensor([smiles])
            
            # 保存特征成.pt文件
            data = Data(smiles_graph = smiles_graph, pro_graph = pro_graph, pro_A = pro_A, target = target, smiles = smiles, 
                        ecfp4 = ecfp4, smile = smile, sequence = sequence, target_id =target_id, drug_id = drug_id, y = label)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data,slices),self.processed_paths[0]) 


if __name__ == '__main__':
    train_root = '/MM-IDTarget/data/train'
    test_root = '/MM-IDTarget/data/test'
    
    train_dataset = SMILES_Protein_Dataset(root=train_root,raw_dataset='train_data90_clusters.csv',processed_data='train_data90_clusters.pt')
    test_dataset = SMILES_Protein_Dataset(root=test_root,raw_dataset='test_data10_clusters.csv',processed_data='test_data10_clusters.pt')
