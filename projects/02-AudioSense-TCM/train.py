"""
AudioSense-TCM 训练脚本
使用合成数据训练VAE
"""
import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path

from vae_model import AudioVAE, ConditionalAudioVAE

def generate_synthetic_data(n_samples=1000, seq_len=128, num_classes=5):
    """
    生成合成语音特征数据
    
    Args:
        n_samples: 样本数量
        seq_len: 特征维度
        num_classes: 类别数量（对应5个脏腑）
    
    Returns:
        X: 特征数据 (n_samples, seq_len)
        labels: 标签 (n_samples,)
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    X = np.random.randn(n_samples, seq_len).astype(np.float32)
    
    # 为不同类别添加不同的偏移，使潜空间更易分离
    for c in range(num_classes):
        start_idx = c * (n_samples // num_classes)
        end_idx = (c + 1) * (n_samples // num_classes) if c < num_classes - 1 else n_samples
        
        # 添加类别特定的偏移
        offset = np.zeros(seq_len)
        offset[c * (seq_len // num_classes):] = 0.5  # 类别特征
        
        X[start_idx:end_idx] += offset
    
    # 5个类别对应5个主要脏腑
    labels = np.random.randint(0, num_classes, size=n_samples)
    
    return torch.from_numpy(X), torch.from_numpy(labels)


def train_standard_vae(epochs=50, batch_size=64, lr=1e-3, save_path='/tmp/audiosense_vae.pth'):
    """训练标准VAE"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = AudioVAE(input_dim=128, hidden_dim=256, latent_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 生成数据
    X, labels = generate_synthetic_data(n_samples=2000, seq_len=128)
    X, labels = X.to(device), labels.to(device)
    
    print(f"数据形状: X={X.shape}, labels={labels.shape}")
    
    # 训练循环
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        recon_losses = []
        kl_losses = []
        cls_losses = []
        
        # Mini-batch训练
        indices = torch.randperm(len(X))
        for i in range(0, len(X), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_x = X[batch_idx]
            batch_labels = labels[batch_idx]
            
            optimizer.zero_grad()
            
            # 前向传播
            recon, mu, logvar, zangfu_logits = model(batch_x)
            
            # 计算损失
            loss, loss_dict = model.loss_function(
                recon, batch_x, mu, logvar, zangfu_logits, batch_labels, beta=1.0
            )
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            recon_losses.append(loss_dict['recon_loss'])
            kl_losses.append(loss_dict['kl_loss'])
            cls_losses.append(loss_dict['cls_loss'])
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(X)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
                  f"Recon: {np.mean(recon_losses):.4f}, "
                  f"KL: {np.mean(kl_losses):.4f}, "
                  f"CLS: {np.mean(cls_losses):.4f}")
    
    # 保存模型
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': 128,
            'hidden_dim': 256,
            'latent_dim': 32,
            'num_organs': 5
        }
    }, save_path)
    print(f"模型已保存到 {save_path}")
    
    return model


def train_conditional_vae(epochs=50, batch_size=64, lr=1e-3, save_path='/tmp/audiosense_cvae.pth'):
    """训练条件VAE"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = ConditionalAudioVAE(
        input_dim=128, hidden_dim=256, latent_dim=32,
        num_organs=5, num_conditions=10
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 生成数据
    X, labels = generate_synthetic_data(n_samples=2000, seq_len=128)
    conditions = torch.randint(0, 10, (len(X),))  # 随机条件
    
    X, labels, conditions = X.to(device), labels.to(device), conditions.to(device)
    
    print(f"数据形状: X={X.shape}, labels={labels.shape}, conditions={conditions.shape}")
    
    # 训练循环
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        indices = torch.randperm(len(X))
        for i in range(0, len(X), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_x = X[batch_idx]
            batch_labels = labels[batch_idx]
            batch_conditions = conditions[batch_idx]
            
            optimizer.zero_grad()
            
            # 前向传播
            recon, mu, logvar, logits = model(batch_x, batch_conditions)
            
            # 损失
            recon_loss = nn.functional.mse_loss(recon, batch_x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            cls_loss = nn.functional.cross_entropy(logits, batch_labels)
            
            loss = recon_loss + kl_loss + cls_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(X):.4f}")
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': 128,
            'hidden_dim': 256,
            'latent_dim': 32,
            'num_organs': 5,
            'num_conditions': 10
        }
    }, save_path)
    print(f"条件VAE模型已保存到 {save_path}")
    
    return model


def evaluate_model(model, device='cpu'):
    """评估模型"""
    model.eval()
    
    # 生成测试数据
    X_test, labels_test = generate_synthetic_data(n_samples=500, seq_len=128)
    X_test, labels_test = X_test.to(device), labels_test.to(device)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(len(X_test)):
            x = X_test[i:i+1]
            label = labels_test[i:i+1]
            
            _, mu, _, zangfu_logits = model(x)
            probs = torch.softmax(zangfu_logits, dim=1)
            pred = torch.argmax(probs, dim=1)
            
            if pred.item() == label.item():
                correct += 1
            total += 1
    
    accuracy = correct / total
    print(f"测试集准确率: {accuracy:.4f} ({correct}/{total})")
    
    return accuracy


if __name__ == '__main__':
    print("=" * 60)
    print("AudioSense-TCM 模型训练")
    print("=" * 60)
    
    # 训练标准VAE
    print("\n[1] 训练标准VAE...")
    model = train_standard_vae(epochs=50, batch_size=64)
    
    # 评估
    print("\n[2] 评估模型...")
    evaluate_model(model)
    
    print("\n训练完成!")
