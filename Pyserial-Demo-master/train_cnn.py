import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# 定义CNN网络结构
class AngleCNN(nn.Module):
    def __init__(self, input_channels=6):
        super(AngleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # 修改第一层池化为更小的kernel_size
            nn.Conv1d(input_channels, 64, kernel_size=1, padding=0),  # 修改kernel为1
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            # 移除第二层池化
            nn.Conv1d(64, 128, kernel_size=1, padding=0),  # 修改kernel为1
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            # 修改最后一层池化方式
            nn.Conv1d(128, 256, kernel_size=1, padding=0),  # 修改kernel为1
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1)
        )

        # 将全连接层定义移到初始化方法中
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # 删除重复的forward定义
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def train_angle_model():
    # 加载数据
    data = pd.read_csv(r'E:\fight_for_py\CyBergearExoskeleton\DIYbyCybergear-main\motion_control\Pyserial-Demo-master\angle_processed_features.csv')
    features = ['forearm_MAV', 'forearm_RMS', 'forearm_VAR',
               'upperarm_MAV', 'upperarm_RMS', 'upperarm_VAR']
    target = 'joint_angle'

    # 数据预处理
    X = data[features].values.astype(np.float32)
    y = data[target].values.astype(np.float32).reshape(-1, 1)
    
    # 标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # 转换为3D输入 (样本数, 通道数, 序列长度)
    X_3d = X_scaled.reshape(-1, 6, 1)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_3d, y_scaled, test_size=0.2, random_state=42
    )
    
    # 创建DataLoader
    train_data = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_data = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AngleCNN().to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 训练循环
    for epoch in range(500):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 验证
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, labels).item()
        
        print(f'Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}, Test Loss: {test_loss/len(test_loader):.4f}')
    
    # 保存模型和标准化器之后添加验证代码
    # 加载最佳模型
    model.load_state_dict(torch.load('angle_cnn_model.pth'))
    
    # 完整验证流程
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # 反标准化
            preds = scaler_y.inverse_transform(outputs.cpu().numpy())
            trues = scaler_y.inverse_transform(labels.numpy())
            
            all_preds.extend(preds.flatten().tolist())
            all_labels.extend(trues.flatten().tolist())

    # 计算评估指标
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    r2 = r2_score(all_labels, all_preds)
    
    print(f'\nFinal Validation: MAE={mae:.2f}°, RMSE={rmse:.2f}°, R²={r2:.4f}')
    
    # 保存验证结果
    results_df = pd.DataFrame({'True': all_labels, 'Predicted': all_preds})
    results_df.to_csv('cnn_validation_results.csv', index=False)

if __name__ == "__main__":
    train_angle_model()