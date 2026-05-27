"""
Variational Autoencoder for Zang-Fu Function Mapping
变分自编码器：语音特征→脏腑功能状态潜空间
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioVAE(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, latent_dim=32, num_organs=5):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Zang-Fu mapping head (5 organs: 心肝脾肺肾)
        self.zangfu_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_organs)
        )
        
        self.latent_dim = latent_dim
        self.num_organs = num_organs
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        zangfu_logits = self.zangfu_head(z)
        return recon, mu, logvar, zangfu_logits
    
    def loss_function(self, recon, x, mu, logvar, zangfu_logits, labels, beta=1.0):
        """
        VAE loss + 分类loss
        
        Args:
            recon: 重构输出
            x: 原始输入
            mu: 潜变量均值
            logvar: 潜变量对数方差
            zangfu_logits: 脏腑分类logits
            labels: 脏腑分类标签
            beta: KL散度权重 (β-VAE)
        
        Returns:
            total_loss: 总损失
            loss_dict: 损失分解
        """
        # 重构损失 (MSE)
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        
        # KL散度
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 分类损失 (CrossEntropy)
        cls_loss = F.cross_entropy(zangfu_logits, labels)
        
        # 总损失
        total_loss = recon_loss + beta * kl_loss + cls_loss
        
        loss_dict = {
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'cls_loss': cls_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def get_zangfu_probs(self, z):
        """从潜变量获取脏腑概率"""
        with torch.no_grad():
            logits = self.zangfu_head(z)
            probs = F.softmax(logits, dim=-1)
        return probs
    
    def infer_latent(self, x):
        """推理潜变量"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class ConditionalAudioVAE(nn.Module):
    """
    条件VAE，用于在给定证候条件下生成语音特征
    """
    def __init__(self, input_dim=128, hidden_dim=256, latent_dim=32, num_organs=5, num_conditions=10):
        super().__init__()
        
        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Embedding(num_conditions, 32),
            nn.Linear(32, 64)
        )
        
        # 联合编码器
        self.joint_encoder = nn.Sequential(
            nn.Linear(input_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 分类头
        self.classifier = nn.Linear(latent_dim, num_organs)
        
        self.latent_dim = latent_dim
    
    def encode(self, x, condition):
        """编码带条件"""
        cond_emb = self.condition_encoder(condition)
        joint = torch.cat([x, cond_emb], dim=-1)
        h = self.joint_encoder(joint)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, condition):
        """解码带条件"""
        cond_emb = self.condition_encoder(condition)
        joint = torch.cat([z, cond_emb], dim=-1)
        return self.decoder(joint)
    
    def forward(self, x, condition):
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, condition)
        logits = self.classifier(z)
        return recon, mu, logvar, logits


class LatentSpaceClassifier(nn.Module):
    """
    潜空间分类器 - 用于在潜空间中进行脏腑状态分类
    """
    def __init__(self, latent_dim=32, num_organs=5, hidden_dim=64):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_organs)
        )
    
    def forward(self, z):
        return self.classifier(z)


if __name__ == '__main__':
    # 测试代码
    batch_size = 4
    input_dim = 128
    latent_dim = 32
    num_organs = 5
    
    # 随机输入
    x = torch.randn(batch_size, input_dim)
    labels = torch.randint(0, num_organs, (batch_size,))
    
    # 测试标准VAE
    print("=" * 40)
    print("AudioVAE 测试")
    print("=" * 40)
    
    model = AudioVAE(input_dim=input_dim, hidden_dim=256, latent_dim=latent_dim)
    recon, mu, logvar, zangfu_logits = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"重构输出形状: {recon.shape}")
    print(f"潜变量均值形状: {mu.shape}")
    print(f"潜变量对数方差形状: {logvar.shape}")
    print(f"脏腑分类logits形状: {zangfu_logits.shape}")
    
    # 计算损失
    loss, loss_dict = model.loss_function(recon, x, mu, logvar, zangfu_logits, labels)
    print(f"\n损失分解:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
    
    # 获取脏腑概率
    z = mu  # 使用均值作为潜变量
    probs = model.get_zangfu_probs(z)
    print(f"\n脏腑概率 (batch={batch_size}, organs={num_organs}):")
    print(probs)
    
    # 测试条件VAE
    print("\n" + "=" * 40)
    print("ConditionalAudioVAE 测试")
    print("=" * 40)
    
    cond_model = ConditionalAudioVAE(input_dim=input_dim, hidden_dim=256, latent_dim=latent_dim)
    condition = torch.randint(0, 10, (batch_size,))
    
    cond_recon, cond_mu, cond_logvar, cond_logits = cond_model(x, condition)
    print(f"条件VAE重构输出形状: {cond_recon.shape}")
    print(f"条件VAE潜变量均值形状: {cond_mu.shape}")
    
    print("\n测试完成!")
