"""
Five Elements Constraint for VQC
五行相生相克约束应用于量子门参数
"""
class FiveElementsConstraint:
    """
    五行: 木火土金水
    相生: 木→火→土→金→水→木
    相克: 木克土克水克火克金克木
    """
    WUXING = ['木', '火', '土', '金', '水']
    SHENG = {'木': '火', '火': '土', '土': '金', '金': '水', '水': '木'}
    KE = {'木': '土', '土': '水', '水': '火', '火': '金', '金': '木'}
    
    def apply_constraint(self, thetas, organ_probs):
        """
        应用五行约束修正参数
        如果某证候被抑制，则降低对应量子比特旋转角度
        """
        modified_thetas = thetas.copy()
        
        # 找出激活最强的证候
        max_org = max(organ_probs, key=organ_probs.get)
        if max_org not in self.WUXING:
            return modified_thetas
        
        idx = self.WUXING.index(max_org)
        
        # 增强相生方向的参数
        sheng_target = self.SHENG[max_org]
        if sheng_target in organ_probs and organ_probs[sheng_target] > 0:
            sheng_idx = self.WUXING.index(sheng_target)
            if sheng_idx * 2 < len(modified_thetas):
                modified_thetas[sheng_idx * 2] *= 1.2
        
        # 抑制相克方向的参数
        ke_target = self.KE[max_org]
        if ke_target in organ_probs and organ_probs[ke_target] > 0.1:
            ke_idx = self.WUXING.index(ke_target)
            if ke_idx * 2 < len(modified_thetas):
                modified_thetas[ke_idx * 2] *= 0.5
        
        return modified_thetas
    
    def validate_transition(self, from_syndrome, to_syndrome):
        """验证证候转移是否满足五行约束"""
        if from_syndrome not in self.WUXING or to_syndrome not in self.WUXING:
            return True
        
        # 检查相生
        if self.SHENG.get(from_syndrome) == to_syndrome:
            return True
        # 检查相克
        if self.KE.get(from_syndrome) == to_syndrome:
            return True
        # 同一行也可以（不变）
        if from_syndrome == to_syndrome:
            return True
        
        return False  # 违反约束
