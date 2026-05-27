"""
Jun-Chen-Zuo-Shi Attention Mechanism
将药性权重嵌入Transformer注意力头
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class JunChenZuoShiAttention(nn.Module):
    """
    君(主药): weight 0.5
    臣(辅药): weight 0.3
    佐(佐药): weight 0.15
    使(使药): weight 0.05
    
    这是一个创新模块，将中医君臣佐使配伍理论嵌入到Transformer注意力机制中
    """
    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()
        self.role_weights = {
            'jun': 0.5,
            'chen': 0.3,
            'zuo': 0.15,
            'shi': 0.05
        }
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Role embedding for herbs
        self.role_embedding = nn.Embedding(4, embed_dim)  # 4 roles: jun, chen, zuo, shi
        
        # 可学习的角色权重调整（soft权重）
        self.role_weight_logits = nn.Parameter(torch.zeros(4))
    
    def forward(self, herbs, role_labels, query):
        """
        Args:
            herbs: (batch, seq, embed) - herb embeddings
            role_labels: (batch, seq) - role indices (0=jun, 1=chen, 2=zuo, 3=shi)
            query: (batch, query_len, embed) - query vector
        
        Returns:
            output: (batch, query_len, embed) - attention weighted representations
            attention_weights: (batch, num_heads, query_len, seq) - attention matrices
        """
        batch_size, herb_seq_len, _ = herbs.shape
        _, query_len, _ = query.shape
        
        # 获取角色嵌入
        role_emb = self.role_embedding(role_labels)  # (batch, seq, embed)
        
        # 计算软角色权重
        soft_weights = F.softmax(self.role_weight_logits, dim=0)
        role_scale = torch.zeros(batch_size, herb_seq_len, self.embed_dim, device=herbs.device)
        for i, (role_name, base_weight) in enumerate(self.role_weights.items()):
            mask = (role_labels == i).float().unsqueeze(-1)  # (batch, seq, 1)
            role_scale += mask * (base_weight * soft_weights[i] * self.embed_dim)
        
        # 调整herb表示
        herbs_adjusted = herbs + role_emb + role_scale * 0.1  # 残差连接
        
        # 计算Q, K, V
        Q = self.q_proj(query)  # (batch, query_len, embed)
        K = self.k_proj(herbs_adjusted)  # (batch, herb_seq, embed)
        V = self.v_proj(herbs_adjusted)  # (batch, herb_seq, embed)
        
        # 多头分割
        Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, herb_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, herb_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # 应用角色权重到注意力分数
        role_weight_matrix = self._get_role_weight_matrix(role_labels, herbs.device)  # (batch, herb_seq, herb_seq)
        role_weight_matrix = role_weight_matrix.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attention_scores = attention_scores + role_weight_matrix
        
        # Softmax归一化
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 应用注意力到值
        context = torch.matmul(attention_weights, V)  # (batch, num_heads, query_len, head_dim)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, query_len, self.embed_dim)
        output = self.out_proj(context)
        
        return output, attention_weights
    
    def _get_role_weight_matrix(self, role_labels, device):
        """
        创建基于角色的权重矩阵
        用于在注意力计算中引入君臣佐使约束
        """
        batch_size, seq_len = role_labels.shape
        weight_matrix = torch.zeros(batch_size, seq_len, seq_len, device=device)
        
        for b in range(batch_size):
            for i in range(seq_len):
                for j in range(seq_len):
                    role_i = role_labels[b, i].item()
                    role_j = role_labels[b, j].item()
                    
                    # 君药对所有药物有更强的注意力
                    if role_i == 0:  # jun
                        if role_j == 0:  # jun-jun
                            weight_matrix[b, i, j] = 0.5
                        elif role_j == 1:  # jun-chen
                            weight_matrix[b, i, j] = 0.3
                        else:
                            weight_matrix[b, i, j] = 0.1
                    # 臣药主要关注君药
                    elif role_i == 1:  # chen
                        if role_j == 0:  # chen-jun
                            weight_matrix[b, i, j] = 0.4
                        elif role_j == 1:  # chen-chen
                            weight_matrix[b, i, j] = 0.25
                        else:
                            weight_matrix[b, i, j] = 0.1
                    # 佐药关注臣药
                    elif role_i == 2:  # zuo
                        if role_j == 1:  # zuo-chen
                            weight_matrix[b, i, j] = 0.3
                        elif role_j == 2:  # zuo-zuo
                            weight_matrix[b, i, j] = 0.15
                        else:
                            weight_matrix[b, i, j] = 0.1
                    # 使药关注佐药
                    else:  # shi
                        if role_j == 2:  # shi-zuo
                            weight_matrix[b, i, j] = 0.2
                        elif role_j == 3:  # shi-shi
                            weight_matrix[b, i, j] = 0.05
                        else:
                            weight_matrix[b, i, j] = 0.1
        
        return weight_matrix
    
    def get_role_weights_summary(self):
        """返回当前学习的角色权重"""
        soft_weights = F.softmax(self.role_weight_logits, dim=0)
        summary = {}
        role_names = ['jun', 'chen', 'zuo', 'shi']
        for i, name in enumerate(role_names):
            base = self.role_weights[name]
            learned = soft_weights[i].item()
            summary[name] = {
                'base_weight': base,
                'learned_adjustment': learned,
                'final_weight': base * learned * self.embed_dim
            }
        return summary


class JunChenZuoShiTransformer(nn.Module):
    """
    完整的Transformer编码器，集成了君臣佐使注意力机制
    """
    def __init__(self, vocab_size=1000, embed_dim=256, num_heads=4, num_layers=2, ff_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, embed_dim) * 0.1)
        
        self.attention_layers = nn.ModuleList([
            JunChenZuoShiAttention(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        self.classifier = nn.Linear(embed_dim, 5)  # 5个证候分类
    
    def forward(self, herb_ids, role_labels, query_ids=None):
        """
        Args:
            herb_ids: (batch, herb_seq) - herb token IDs
            role_labels: (batch, herb_seq) - role indices
            query_ids: (batch, query_seq) - query token IDs, if None use herb_ids
        
        Returns:
            logits: (batch, num_classes) - syndrome classification logits
        """
        batch_size = herb_ids.shape[0]
        
        # 获取herb embeddings
        herbs = self.embedding(herb_ids) + self.pos_encoding[:, :herb_ids.shape[1], :]
        
        # 获取query embeddings
        if query_ids is None:
            query = herbs
        else:
            query = self.embedding(query_ids) + self.pos_encoding[:, :query_ids.shape[1], :]
        
        # 通过注意力层
        x = herbs
        for attn_layer in self.attention_layers:
            attn_out, _ = attn_layer(x, role_labels, query)
            x = self.layer_norm1(x + attn_out)
            ff_out = self.feed_forward(x)
            x = self.layer_norm2(x + ff_out)
        
        # 使用query位置的表示进行分类
        pooled = x.mean(dim=1)  # (batch, embed_dim)
        logits = self.classifier(pooled)
        
        return logits


if __name__ == '__main__':
    # 测试代码
    batch_size = 2
    herb_seq_len = 10
    query_len = 5
    embed_dim = 64
    num_heads = 4
    
    # 随机输入
    herb_ids = torch.randint(0, 100, (batch_size, herb_seq_len))
    role_labels = torch.randint(0, 4, (batch_size, herb_seq_len))  # 0=jun, 1=chen, 2=zuo, 3=shi
    query_ids = torch.randint(0, 100, (batch_size, query_len))
    
    # 测试注意力模块
    attn = JunChenZuoShiAttention(embed_dim, num_heads)
    herbs = torch.randn(batch_size, herb_seq_len, embed_dim)
    query = torch.randn(batch_size, query_len, embed_dim)
    
    output, attn_weights = attn(herbs, role_labels, query)
    print(f"Attention output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # 测试完整Transformer
    transformer = JunChenZuoShiTransformer(vocab_size=100, embed_dim=embed_dim, num_heads=num_heads, num_layers=2)
    logits = transformer(herb_ids, role_labels, query_ids)
    print(f"Transformer output shape: {logits.shape}")
    
    # 打印角色权重
    print("\n角色权重:")
    for role, info in attn.get_role_weights_summary().items():
        print(f"  {role}: base={info['base_weight']:.2f}, learned={info['learned_adjustment']:.4f}")
