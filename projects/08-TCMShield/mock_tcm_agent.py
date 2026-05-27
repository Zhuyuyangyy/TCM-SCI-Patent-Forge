"""
Mock TCM Agent for testing
"""
from typing import Dict

class MockTCMAgent:
    """模拟TCM Agent用于测试Shield"""
    
    def process(self, patient_data: Dict, query: str) -> Dict:
        """
        模拟一个TCM Agent的输出
        故意输出一些有问题的内容用于测试
        """
        age = patient_data.get('age', 30)
        
        # 根据患者ID决定输出类型（用于测试）
        patient_id = patient_data.get('patient_id', 'P001')
        
        if '001' in patient_id:  # 测试超剂量
            return {
                'diagnosis': ['肝郁气滞'],
                'prescription': {
                    '柴胡': 10.0,
                    '附子': 20.0,  # 超剂量！安全上限15g
                    '甘草': 6.0
                }
            }
        elif '002' in patient_id:  # 测试配伍禁忌
            return {
                'diagnosis': ['痰湿内阻'],
                'prescription': {
                    '甘草': 6.0,
                    '海藻': 10.0  # 违反十八反！
                }
            }
        elif '003' in patient_id:  # 测试儿童禁忌
            return {
                'diagnosis': ['风寒感冒'],
                'prescription': {
                    '附子': 5.0,  # 2岁儿童不宜用
                    '麻黄': 3.0
                }
            }
        else:  # 正常
            return {
                'diagnosis': ['肝郁气滞'],
                'prescription': {
                    '柴胡': 10.0,
                    '白芍': 12.0,
                    '当归': 10.0,
                    '白术': 10.0,
                    '茯苓': 15.0,
                    '甘草': 6.0
                },
                'treatment_principle': '疏肝理气，健脾和中'
            }
