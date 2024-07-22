from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
import torch
import os
import dgl
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score
from src.models.MM_IDTarget import MM_IDTarget
    

class SMILES_Protein_Dataset(InMemoryDataset):
    def __init__(self, root, raw_dataset=None, processed_data=None, transform = None, pre_transform = None):
        self.root=root
        self.raw_dataset=raw_dataset
        self.processed_data=processed_data
        self.max_smiles_len=256
        self.smiles_dict_len=65
        super(SMILES_Protein_Dataset,self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0]) 
        
        
#训练模型
def train():
    model.train()
    total_loss = 0
    for idx,data in loop:
        # 读取 ProA.csv 文件
        import torch.nn as nn
        loss_fn = nn.CrossEntropyLoss()
        out = model(dgl.batch(data.smiles_graph).to(device), dgl.batch(data.pro_graph).to(device), data.pro_A.to(device), data.target.to(device), data.smiles.to(device), data.ecfp4.to(device))
        # 将输入数据传入模型进行前向传播
        loss = loss_fn(out, data.y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loop)


# 测试模型，并计算Top1-10指标
def test(loader):
    model.eval()
    true_labels = []
    predicted_labels = []
    predicted_probs = []
    with torch.no_grad():
        for idx,data in loader:
            optimizer.zero_grad()
            out = model(dgl.batch(data.smiles_graph).to(device), dgl.batch(data.pro_graph).to(device), data.pro_A.to(device), data.target.to(device), data.smiles.to(device), data.ecfp4.to(device))
            pred1 = out.argmax(dim=1)
            true_labels.extend(data.y.tolist())
            predicted_labels.extend(pred1.tolist())
            predicted_probs.extend(out.tolist())
    # 计算SP和SE
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    predicted_probs = np.array(predicted_probs)
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).flatten()
    print(tn, fp, fn, tp)
    sp = tn / (tn + fp)
    se = tp / (tp + fn)
    # 计算ACC、AUC
    acc = accuracy_score(true_labels, predicted_labels)
    auc = roc_auc_score(true_labels, predicted_probs[:, 1])
    return acc, sp, se, auc


if __name__ == '__main__':
    train_root = '/MM-IDTarget/data/train'
    test_root = '/MM-IDTarget/data/test'
    
    train_dataset = SMILES_Protein_Dataset(root=train_root,raw_dataset='train_data90_clusters.csv',processed_data='train_data90_clusters.pt')
    test_dataset = SMILES_Protein_Dataset(root=test_root,raw_dataset='test_data10_clusters.csv',processed_data='test_data10_clusters.pt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    epochs = 200
    
    # 实例化模型并定义优化器
    model = MM_IDTarget(3, 25 + 1, embedding_size=20).to(device)
    lr=0.0001
    weight_decay=0.0005
    batch_size = 64
    min = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 打开文件，以追加方式写入
    for epoch in range(epochs):
        print("Epoch: " + str(epoch+1))
        train_dataloader = DataLoader(train_dataset,batch_size,shuffle=True,drop_last=False)
        test_dataloader = DataLoader(test_dataset,batch_size,shuffle=False,drop_last=False)
        with open(f"/home/user/sgp/Chemgenomic/target/MM-IDTarget/result.txt", 'a') as f:
            
            loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), colour='red', desc='Train Loss')
            loss = train()
            
            loop3 = tqdm(enumerate(test_dataloader), total=len(test_dataloader), colour='red', desc='Test')
            acc3, sp3, se3, auc3 = test(loop3)
            
            print(loss)
            print(acc3, se3, sp3, auc3)
                
            f.write('Epoch {:03d}, Loss: {:.4f}\n'.format(epoch+1, loss))
            
            f.write('Test:  Acc: {:.4f}, SP: {:.4f}, SE: {:.4f}, AUC: {:.4f}\n'
                    .format(acc3, sp3, se3, auc3))
            if loss < min:
                min = loss
                torch.save(model.state_dict(), os.path.join(f"/home/user/sgp/Chemgenomic/target/MM-IDTarget/model.pkl"))   
                   
        # 关闭文件
        f.close()
