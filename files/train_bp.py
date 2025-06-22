import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F  # 新增导入
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

# 在网络架构部分增加层间连接（修改以下部分）
class BPNet(nn.Module):
    def __init__(self, input_size):
        super(BPNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4)
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x))) + x[:, :128]  # 添加残差连接
        x = F.relu(self.fc3(x))
        return self.fc4(x)

def train_emg_model():
    # 加载数据
    data = pd.read_csv(r'E:\fight_for_py\CyBergearExoskeleton\DIYbyCybergear-main\motion_control\Pyserial-Demo-master\motor_intention_features.csv')
    features = ['forearm_MAV', 'forearm_RMS', 'forearm_VAR',
               'upperarm_MAV', 'upperarm_RMS', 'upperarm_VAR']
    target = 'joint_angle'

    # 数据预处理
    X = data[features].values.astype(np.float32)
    y = data[target].values.astype(np.int64).flatten()
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 转换为PyTorch张量
    train_data = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_data = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    
    # 创建DataLoader
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BPNet(X_train.shape[1]).to(device)
    
    # 定义损失函数和优化器
    # 添加类别权重计算
    class_counts = np.bincount(y_train)
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    weights = weights / weights.sum()
    weights = weights.to(device)

    # 修改损失函数
    criterion = nn.CrossEntropyLoss(weight=weights)  # 添加类别权重
    
    # 修改优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # 使用AdamW+权重衰减
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.5)
    
    # 调整早停参数
    patience = 30  # 增加耐心值
    best_acc = 0
    no_improve = 0

    # 修改训练循环
    for epoch in range(200):  # 增加最大epoch
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = correct / total
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.4f}')
        
        # 早停机制
        if acc > best_acc:
            best_acc = acc
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    
    # 最终评估
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    print(classification_report(all_labels, all_preds))
    
    # 保存标准化器
    torch.save({
        'scaler_mean': torch.tensor(scaler.mean_),
        'scaler_scale': torch.tensor(scaler.scale_)
    }, 'scaler_params.pth')

if __name__ == "__main__":
    train_emg_model()