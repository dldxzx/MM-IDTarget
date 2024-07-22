import torch
import pandas as pd
from rdkit import Chem
import os
from src.models.MM_IDTarget_Recall import MM_IDTarget
from src.DrugGraph import smiles_to_graph
from rdkit.Chem import AllChem
import logging
import numpy as np
from DeepPurpose import utils
from DeepPurpose import DTI as models
import warnings
warnings.filterwarnings("ignore")    

# 设置设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
Models = MM_IDTarget(3, 25 + 1, embedding_size=128).to(device)  

VOCAB_PROTEIN = {"A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6,
                 "H": 7, "I": 8, "K": 9, "L": 10, "M": 11, "N": 12,
                 "P": 13, "Q": 14, "R": 15, "S": 16, "T": 17, "V": 18,
                 "W": 19, "Y": 20, "X": 21}


def seqs2int(target):
    return [VOCAB_PROTEIN[s] for s in target]

VOCAB_SMILES = {
    'C': 1, 'c': 2, 'N': 3, 'n': 4, 'O': 5, 'o': 6, 'H': 7, 'h': 8, 'P': 9, 'p': 10, 'S': 11, 's': 12, 'F': 13, 'f': 14,
    'Cl': 15, 'Br': 16, 'I': 17, 'B': 18, 'b': 19, 'K': 20, 'k': 21, 'V': 22, 'v': 23, 'Mg': 24, 'm': 25, 'Li': 26, 'l': 27,
    'Zn': 28, 'z': 29, 'Si': 30, 'i': 31, 'As': 32, 'a': 33, 'Fe': 34, 'e': 35, 'Cu': 36, 'u': 37, 'Se': 38, 'se': 39,
    'R': 40, 'r': 41, 'T': 42, 't': 43, 'Na': 44, 'a': 45,'G': 46, 'g': 47, 'D': 48, 'd': 49, 'Q': 50, 'q': 51, 'X': 52, 
    'x': 53,'*': 54, '#': 55, '1': 56, '2': 57, '3': 58, '4': 59, '5': 60, '6': 61, '7': 62, '8': 63, '9': 64, '0': 65,
    '(': 66, ')': 67, '[': 68, ']': 69, '=': 70, '/': 71, '\\': 72, '@': 73, '+': 74, '-': 75, '.':76}

# 反向字典
NUM_TO_LETTER = {v: k for k, v in VOCAB_SMILES.items()}

def smi2int(smile):
    # 将 SMILES 字符串转换成整数序列
    return [VOCAB_SMILES[s] for s in smile]

# 定义氨基酸序列的字母表
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

def seq_to_one_hot(seq):
    # 创建一个全为0的数组，形状为 (序列长度, 字母表大小)
    one_hot = np.zeros((len(seq), len(amino_acids)))
    
    # 使用字典将氨基酸映射到索引
    aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}
    
    # 将序列中的每个氨基酸转换为 one-hot 编码
    for i, aa in enumerate(seq):
        if aa in aa_to_index:
            one_hot[i, aa_to_index[aa]] = 1 
    return one_hot

# 定义所有可能的字符列表
all_characters = ['C', 'c', 'N', 'n', 'O', 'o', 'H', 'h', 'P', 'p', 'S', 's', 'F', 'f', 'Cl', 'Br', 'I', 'B', 'b', 
                  'K', 'k', 'V', 'v', 'Mg', 'm', 'Li', 'l', 'Zn', 'z', 'Si', 'i', 'As', 'a', 'Fe', 'e', 'Cu', 'u', 
                  'Se', 'se', 'R', 'r', 'T', 't', 'Na', 'a', 'G', 'g', 'D', 'd', 'Q', 'q', 'X', 'x', '*', '#', '1', 
                  '2', '3', '4', '5', '6', '7', '8', '9', '0', '(', ')', '[', ']', '=', '/', '\\', '@', '+', '-', '.']

# 定义氨基酸序列的字母表
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'


# 定义函数来计算分子指纹
def calculate_fingerprints(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        ecfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)  # 计算ECFP4指纹
        ecfp4_array = [int(x) for x in ecfp4.ToBitString()]  # 将ECFP4指纹转换为0和1组成的数组
        return ecfp4_array
    else:
        return None
    

def SMILES_Protein_Dataset(sequence, smile, target_id):
    ecfp4_array = calculate_fingerprints(smile)
    ecfp4 = torch.tensor(ecfp4_array).unsqueeze(0)
    df_proA = pd.read_csv("/MM-IDTarget/data/ProA.csv")
    row_data_A = df_proA[df_proA['target_id'] == target_id]
    row_data_list_A = row_data_A.values.tolist()[0][1:]
    pro_A = torch.tensor(row_data_list_A).unsqueeze(0)
    target = seqs2int(sequence)
    target_len = 1200
    if len(target) < target_len:
        target = np.pad(target, (0, target_len - len(target)),mode='constant')
    else:
        target = target[:target_len]
    target=torch.LongTensor([target])
    smiles_graph = smiles_to_graph(smile)
    pro_graph = torch.load(f'/MM-IDTarget/data/protein_EW-GCN/{target_id}.pt')
    smiles = smi2int(smile)
    if len(smiles) < 174:
        smiles = np.pad(smiles, (0, 174 - len(smiles)),mode='constant')
    smiles = torch.LongTensor([smiles])

    with torch.no_grad():
        out = Models(smiles_graph.to(device), pro_graph.to(device), pro_A.to(device), target.to(device), smiles.to(device), ecfp4.to(device))
        second_column = out[:, 1]
    return second_column


def process_data_function(sorted_targets, matching_rows, drug_encoding, target_encoding, ff2_modelss, smiles, target):
    matching_rows = [(index, row) for index, row in enumerate(sorted_targets) if row[1] == target]
    filtered_rows = [row for row in sorted_targets if row[2] > matching_rows[0][1][2]]
    data_DTA = [row for row in sorted_targets if row[2] == matching_rows[0][1][2]]   
    ff2_DTA_list_all = []
    for data_dta in data_DTA:  
        ff2_X_pred = utils.data_process([smiles], [data_dta[1]], [0], 
                    drug_encoding, target_encoding, 
                    split_method='no_split')
        ff2_DTA_sorce = ff2_modelss.predict(ff2_X_pred)    
        ff2_DTA_list_all.append((data_dta[0], data_dta[1], ff2_DTA_sorce[0])) 
        if data_dta[1] == target:
            ff2_matching_rows2_all = ff2_DTA_sorce[0]
    ff2_sorted_targets_dta_all = sorted([x[2] for x in ff2_DTA_list_all], reverse=True)
    ff2_filtered_rows21_all = [row for row in ff2_sorted_targets_dta_all if row > ff2_matching_rows2_all]
    ff2_indexs  = len(ff2_filtered_rows21_all) + len(filtered_rows)

    return ff2_indexs
    
            
def Recall_Top859(pretrained_model, target, smiles, target_id):
    # 加载保存好的模型
    if os.path.exists(pretrained_model) and pretrained_model != " ":
        logging.info(f"开始加载MMGAT模型")
        try:
            state_dict = torch.load(pretrained_model)
            new_model_state_dict = Models.state_dict()
            for key in new_model_state_dict.keys():
                if key in state_dict.keys():
                    try:
                        new_model_state_dict[key].copy_(state_dict[key])
                    except:
                        None
            Models.load_state_dict(new_model_state_dict)
            logging.info("MMGAT模型加载成功(!)")
        except:
            new_state_dict = {}
            for key, value in state_dict.items():
                new_state_dict[key.replace("module.", "")] = value
            Models.load_state_dict(new_state_dict)
            logging.info("MMGAT模型加载成功(!!!)")
    else:
        logging.info("模型路径不存在，不能加载模型")
    Models.eval()
    # 读取 CSV 文件
    df1 = pd.read_csv('/MM-IDTarget/data/target.csv')  
    try:
        # 尝试访问DataFrame中的数据
        target_id = df1.loc[df1['sequence'] == target]['target_id'].values[0]
        # 继续处理数据
    except IndexError:
        print("error")
        # 如果发生索引错误，跳过当前循环
    target_list = []
    for j in range(1, 860):
        # 读取 CSV 文件
        df1 = pd.read_csv('/MM-IDTarget/data/target.csv')
        # 动态生成变量名
        target_id = f"T{j}"
        # 在 DataFrame 中查找对应行
        row1 = df1.loc[df1['target_id'] == target_id]
        # 获取 sequence 值
        sequence = row1['sequence'].values[0]
        data = SMILES_Protein_Dataset(sequence, smiles, target_id)
        target_list.append((target_id, sequence, data.item()))
    sorted_targets = sorted(target_list, key=lambda x: x[2], reverse=True)
    matching_rows = [(index, row) for index, row in enumerate(sorted_targets) if row[1] == target]
    if len(matching_rows) != 0:
        filtered_sorted_targets = [(row[0], row[1], round(row[2], 2)) for row in sorted_targets[:next((index for index, row in enumerate(sorted_targets) if row[2] < matching_rows[0][1][2]), len(sorted_targets))]]
        drug_encoding, target_encoding = 'MPNN', 'CNN'
        modelss = models.model_pretrained(path_dir = f'/MM-IDTarget/reslut_model/model_MPNN_CNN') 
        indexs = process_data_function(filtered_sorted_targets, matching_rows, drug_encoding, target_encoding, modelss, smiles, target)
        
    return indexs + 1

target = 'MARRCGPVALLLGFGLLRLCSGVWGTDTEERLVEHLLDPSRYNKLIRPATNGSELVTVQLMVSLAQLISVHEREQIMTTNVWLTQEWEDYRLTWKPEEFDNMKKVRLPSKHIWLPDVVLYNNADGMYEVSFYSNAVVSYDGSIFWLPPAIYKSACKIEVKHFPFDQQNCTMKFRSWTYDRTEIDLVLKSEVASLDDFTPSGEWDIVALPGRRNENPDDSTYVDITYDFIIRRKPLFYTINLIIPCVLITSLAILVFYLPSDCGEKMTLCISVLLALTVFLLLISKIVPPTSLDVPLVGKYLMFTMVLVTFSIVTSVCVLNVHHRSPTTHTMAPWVKVVFLEKLPALLFMQQPRHHCARQRLRLRRRQREREGAGALFFREAPGADSCTCFVNRASVQGLAGAFGAEPAPVAGPGRSGEPCGCGLREAVDGVRFIADHMRSEDDDQSVSEDWKYVAMVIDRLFLWIFVFVCVFGTIGMFLQPLFQNYTTTTFLHSDHSAPSSK'
smiles = 'N1C[C@@H]2C[C@H](C1)c1c2cc2c(c1)nccn2'
target_id = 'T633'
index = Recall_Top859(f"/MM-IDTarget/reslut_model/MM-IDTarget_model.pkl", target, smiles, target_id)
print(index)
