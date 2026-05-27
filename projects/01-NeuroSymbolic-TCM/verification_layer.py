"""
Five Elements Verification Layer
五行生克规则硬约束验证层
"""
from collections import defaultdict

class FiveElementsVerifier:
    WUXING_CYCLE = ['木', '火', '土', '金', '水']
    SHENG = {'木': '火', '火': '土', '土': '金', '金': '水', '水': '木'}  # 相生关系
    KE = {'木': '土', '土': '水', '水': '火', '火': '金', '金': '木'}      # 相克关系
    
    # 五脏归属
    ZANGFU_WUXING = {
        '肝': '木',
        '心': '火',
        '脾': '土',
        '肺': '金',
        '肾': '水',
        '胆': '木',
        '小肠': '火',
        '胃': '土',
        '大肠': '金',
        '膀胱': '水'
    }
    
    def __init__(self):
        self.transition_history = []
    
    def verify_transition(self, from_syndrome, to_syndrome):
        """
        验证证候转移是否满足五行约束
        
        Args:
            from_syndrome: 源证候名称
            to_syndrome: 目标证候名称
        
        Returns:
            (bool, str): (是否通过, 原因说明)
        """
        # 提取证候的五行属性
        from_wuxing = self._get_wuxing_from_syndrome(from_syndrome)
        to_wuxing = self._get_wuxing_from_syndrome(to_syndrome)
        
        if from_wuxing is None or to_wuxing is None:
            # 如果无法确定五行，保守返回通过
            return True, "无法确定五行属性，保守通过"
        
        # 检查相生关系
        if self.SHENG.get(from_wuxing) == to_wuxing:
            reason = f"{from_wuxing}→{to_wuxing}，相生关系（顺生），允许"
            self.transition_history.append((from_syndrome, to_syndrome, 'pass', reason))
            return True, reason
        
        # 检查相克关系
        if self.KE.get(from_wuxing) == to_wuxing:
            reason = f"{from_wuxing}→{to_wuxing}，相克关系（隔一克），允许但需注意"
            self.transition_history.append((from_syndrome, to_syndrome, 'warning', reason))
            return True, reason
        
        # 检查同属
        if from_wuxing == to_wuxing:
            reason = f"{from_wuxing}→{from_wuxing}，同属一行，可能为兼证或演变"
            self.transition_history.append((from_syndrome, to_syndrome, 'warning', reason))
            return True, reason
        
        # 检查被相生（逆向）
        for src, dst in self.SHENG.items():
            if dst == from_wuxing and src == to_wuxing:
                reason = f"{from_wuxing}←{to_wuxing}，被{to_wuxing}所生，逆向相生，需谨慎"
                self.transition_history.append((from_syndrome, to_syndrome, 'warning', reason))
                return True, reason
        
        # 检查被相克（逆向）
        for src, dst in self.KE.items():
            if dst == from_wuxing and src == to_wuxing:
                reason = f"{from_wuxing}←{to_wuxing}，被{to_wuxing}所克，逆向相克，验证不通过"
                self.transition_history.append((from_syndrome, to_syndrome, 'fail', reason))
                return False, reason
        
        # 其他情况：不相生也不相克
        reason = f"{from_wuxing}与{to_wuxing}既不相生也不相克，需人工确认"
        self.transition_history.append((from_syndrome, to_syndrome, 'uncertain', reason))
        return False, reason
    
    def verify_prescription(self, prescription):
        """
        验证处方是否符合五行配伍规则
        
        Args:
            prescription: 处方对象，包含药物列表和角色标签
        
        Returns:
            (bool, list): (是否合规, 警告列表)
        """
        warnings = []
        
        if not hasattr(prescription, 'herbs'):
            return True, warnings
        
        herbs = prescription.herbs if isinstance(prescription.herbs, list) else []
        
        # 检查君臣佐使配伍
        role_count = defaultdict(int)
        for herb in herbs:
            role = getattr(herb, 'role', 'unknown')
            role_count[role] += 1
        
        # 君药必须存在
        if role_count.get('jun', 0) == 0:
            warnings.append("警告：处方中无君药")
        
        # 臣药最好存在
        if role_count.get('chen', 0) == 0:
            warnings.append("建议：处方中无臣药")
        
        # 检查药性冲突
        herb_wuxing = [getattr(h, 'wuxing', None) for h in herbs]
        
        # 同性药物不宜过多
        wuxing_count = defaultdict(int)
        for wx in herb_wuxing:
            if wx:
                wuxing_count[wx] += 1
        
        for wx, count in wuxing_count.items():
            if count > 3:
                warnings.append(f"警告：{wx}属性药物过多（{count}种），可能影响平衡")
        
        is_valid = len([w for w in warnings if '警告' in w]) == 0
        return is_valid, warnings
    
    def suggest_correction(self, invalid_transition):
        """
        对违反五行约束的情况给出修正建议
        
        Args:
            invalid_transition: (from_syndrome, to_syndrome, reason) 元组
        
        Returns:
            list: 修正建议列表
        """
        if len(invalid_transition) < 3:
            return ["无法提供修正建议：信息不足"]
        
        from_syn, to_syn, reason = invalid_transition[0], invalid_transition[1], invalid_transition[2]
        suggestions = []
        
        from_wuxing = self._get_wuxing_from_syndrome(from_syn)
        to_wuxing = self._get_wuxing_from_syndrome(to_syn)
        
        if from_wuxing is None or to_wuxing is None:
            return ["无法确定五行属性，建议咨询中医专家"]
        
        # 找到当前五行在相生链上的下一个
        sheng_next = self.SHENG.get(from_wuxing)
        if sheng_next:
            suggestions.append(f"考虑先治{sheng_next}证，再转{to_syn}")
        
        # 找到当前五行在相克链上的目标
        ke_target = self.KE.get(from_wuxing)
        if ke_target:
            suggestions.append(f"或先调理{ke_target}功能以平衡五行")
        
        # 提供整体调理建议
        suggestions.append(f"建议采用兼顾{from_wuxing}与{to_wuxing}的复方治疗")
        suggestions.append("具体用药请咨询专业中医师")
        
        return suggestions
    
    def _get_wuxing_from_syndrome(self, syndrome):
        """从证候名称推断五行属性"""
        # 直接匹配
        for organ, wx in self.ZANGFU_WUXING.items():
            if organ in syndrome:
                return wx
        
        # 间接推断
        syndrome_lower = syndrome.lower()
        
        # 肝胆属木
        if '肝' in syndrome or '胆' in syndrome:
            return '木'
        
        # 心小肠属火
        if '心' in syndrome or '小肠' in syndrome:
            return '火'
        
        # 脾胃属土
        if '脾' in syndrome or '胃' in syndrome or '湿' in syndrome:
            return '土'
        
        # 肺大肠属金
        if '肺' in syndrome or '大肠' in syndrome or '燥' in syndrome:
            return '金'
        
        # 肾膀胱属水
        if '肾' in syndrome or '膀胱' in syndrome or '寒' in syndrome:
            return '水'
        
        # 常见证候关键词
        if '郁' in syndrome or '风' in syndrome:
            return '木'  # 肝郁
        if '火' in syndrome or '热' in syndrome:
            return '火'  # 心火
        if '虚' in syndrome:
            return '土'  # 脾虚
        if '阴' in syndrome:
            return '水'  # 肾阴虚
        
        return None
    
    def get_transition_history(self):
        """获取转移验证历史"""
        return self.transition_history
    
    def reset_history(self):
        """重置验证历史"""
        self.transition_history = []


if __name__ == '__main__':
    # 测试代码
    verifier = FiveElementsVerifier()
    
    # 测试相生
    ok, reason = verifier.verify_transition('肝郁气滞', '肝郁化火')
    print(f"肝郁气滞→肝郁化火: {'通过' if ok else '不通过'} - {reason}")
    
    # 测试相克
    ok, reason = verifier.verify_transition('肝郁气滞', '脾胃湿热')
    print(f"肝郁气滞→脾胃湿热: {'通过' if ok else '不通过'} - {reason}")
    
    # 测试同属
    ok, reason = verifier.verify_transition('肝郁气滞', '肝阳上亢')
    print(f"肝郁气滞→肝阳上亢: {'通过' if ok else '不通过'} - {reason}")
    
    # 测试修正建议
    print("\n修正建议:")
    for suggestion in verifier.suggest_correction(('肝', '脾', 'invalid')):
        print(f"  - {suggestion}")
