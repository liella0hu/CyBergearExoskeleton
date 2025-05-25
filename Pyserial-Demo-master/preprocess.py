import numpy as np
import pandas as pd
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import numpy
import re

def extract_features(signal):
    """提取时域和频域特征"""
    features = {}
    print("signal: ", signal)
    signal = numpy.array([float(s) for s in signal if not re.search(r'[a-zA-Z]', str(s))])
    # 时域特征
    features['MAV'] = np.mean(np.abs(signal))          # 平均绝对值
    features['RMS'] = np.sqrt(np.mean(signal**2))       # 均方根
    features['VAR'] = np.var(signal)                    # 方差
    features['ZCR'] = len(np.where(np.diff(np.sign(signal)))[0])/len(signal)  # 过零率
    
    # 频域特征
    fft_vals = np.abs(fft(signal))[:50]                # 取前50个FFT系数
    features['MF'] = np.argmax(fft_vals)/len(fft_vals) # 主频占比
    return features

def process_emg_data(input_path, window_size=500, step=50):
    """处理原始EMG数据"""
    df = pd.read_csv(input_path)
    features = []
    
    # 滑动窗口处理
    for i in range(0, len(df)-window_size, step):
        window = df.iloc[i:i+window_size]
        forearm_feat = extract_features(window['forearm_sEMG'].values)
        upperarm_feat = extract_features(window['upperarm_sEMG'].values)
        joint_angle = window['joint_angle'].iloc[-1]
        
        # 合并特征并添加列名前缀
        combined = {
            **{'forearm_' + k: v for k, v in forearm_feat.items()},
            **{'upperarm_' + k: v for k, v in upperarm_feat.items()},
            'joint_angle': joint_angle
        }
        features.append(combined)
    
    # 明确指定列名顺序
    columns = [
        'forearm_MAV', 'forearm_RMS', 'forearm_VAR', 'forearm_ZCR', 'forearm_MF',
        'upperarm_MAV', 'upperarm_RMS', 'upperarm_VAR', 'upperarm_ZCR', 'upperarm_MF',
        'joint_angle'
    ]
    return pd.DataFrame(features, columns=columns)

if __name__ == "__main__":
    # 处理并保存特征数据
    feature_df = process_emg_data(r'E:\fight_for_py\CyBergearExoskeleton\DIYbyCybergear-main\motion_control\Pyserial-Demo-master\angle_processed.csv')
    
    # 拆分特征和标签
    X = feature_df.drop('joint_angle', axis=1)
    y = feature_df['joint_angle']
    
    # 创建标准化器（仅标准化特征）
    scaler = StandardScaler().fit(X.values)
    joblib.dump(scaler, 'feature_scaler.pkl')
    
    # 保存带列名的完整数据
    pd.concat([pd.DataFrame(scaler.transform(X), columns=X.columns), y], axis=1) \
      .to_csv(r'E:\fight_for_py\CyBergearExoskeleton\DIYbyCybergear-main\motion_control\Pyserial-Demo-master\angle_processed_features.csv', index=False)