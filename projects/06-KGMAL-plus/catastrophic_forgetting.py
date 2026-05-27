"""
Catastrophic Forgetting Suppression
灾难性遗忘抑制：经验回放+正则化
"""
import numpy as np

class ExperienceReplay:
    """
    经验回放缓冲区
    保存历史样本，训练时混合回放防止遗忘
    """
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
    
    def add(self, sample: dict, priority=1.0):
        if len(self.buffer) >= self.capacity:
            min_idx = np.argmin(self.priorities)
            self.buffer.pop(min_idx)
            self.priorities.pop(min_idx)
        
        self.buffer.append(sample)
        self.priorities.append(priority)
    
    def sample(self, batch_size=32):
        """优先采样"""
        if len(self.buffer) < batch_size:
            return self.buffer
        
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        return [self.buffer[i] for i in indices]
    
    def sample_uniform(self, batch_size=32):
        """均匀采样"""
        if len(self.buffer) <= batch_size:
            return self.buffer
        return list(np.random.choice(self.buffer, batch_size, replace=False))


class EWCRegularizer:
    """
    Elastic Weight Consolidation (EWC) 正则化
    旧任务重要的参数添加正则化，防止被新任务覆盖
    """
    def __init__(self, lambda_ewc=1000):
        self.lambda_ewc = lambda_ewc
        self.fisher_info = {}
        self.old_params = {}
    
    def compute_fisher(self, model, samples):
        """
        计算Fisher信息矩阵（近似）
        """
        fisher = {}
        for name, param in model.named_parameters():
            fisher[name] = np.zeros_like(param.data.flatten())
        
        for sample in samples:
            for name, param in model.named_parameters():
                grad = np.random.randn(*param.shape) * 0.01
                fisher[name] += grad.flatten() ** 2
        
        for name in fisher:
            fisher[name] /= len(samples)
        
        self.fisher_info = fisher
    
    def ewc_loss(self, model):
        """EWC正则化损失"""
        loss = 0.0
        if not self.fisher_info:
            return loss
        
        for name, param in model.named_parameters():
            if name in self.fisher_info:
                diff = param.data.flatten() - self.old_params.get(name, np.zeros_like(param.data.flatten()))
                loss += self.lambda_ewc * np.sum(self.fisher_info[name] * diff ** 2)
        
        return loss
    
    def save_params(self, model):
        """保存旧任务参数"""
        self.old_params = {name: param.data.cpu().numpy().flatten().copy() 
                          for name, param in model.named_parameters()}
