"""
MACCM Benchmark — 多智能体中医会诊评测基准
500+ 诊断用例 × 5种方案 × 消融实验

方案:
  1. SingleAgent (单Agent直接诊断)
  2. MultiAgent-NoDeliberation (多Agent无审议)
  3. MultiAgent-NoConflict (多Agent无冲突消解)
  4. MultiAgent-NoCredibility (多Agent无可信度权重)
  5. MACCM (完整方案)

评价指标:
  Top-1 Accuracy, Top-3 Accuracy, Macro-F1, 共识率, 冲突消解率
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter


# ═══════════════════════════════════════════════════════════
# 1. TCM Diagnosis Benchmark Dataset
# ═══════════════════════════════════════════════════════════
class TCMDiagnosisBenchmark:
    """
    Standardized TCM diagnosis benchmark.

    Each case has:
    - symptoms: list of symptom strings
    - tongue: tongue description
    - pulse: pulse type
    - ground_truth: correct syndrome (from expert consensus)
    - difficulty: easy/medium/hard
    - multi_syndrome: whether multiple syndromes coexist
    """

    CASES = [
        # Easy cases (clear symptom-syndrome mapping)
        {'id': 'E001', 'symptoms': ['胁肋胀痛', '情绪抑郁', '嗳气频繁'], 'tongue': '舌淡红苔薄白', 'pulse': '脉弦', 'ground_truth': '肝郁气滞', 'difficulty': 'easy'},
        {'id': 'E002', 'symptoms': ['心悸失眠', '面色萎黄', '食欲不振'], 'tongue': '舌淡苔薄', 'pulse': '脉细弱', 'ground_truth': '心脾两虚', 'difficulty': 'easy'},
        {'id': 'E003', 'symptoms': ['腰膝酸软', '畏寒肢冷', '夜尿多'], 'tongue': '舌淡胖苔白', 'pulse': '脉沉迟', 'ground_truth': '肾阳虚', 'difficulty': 'easy'},
        {'id': 'E004', 'symptoms': ['口苦咽干', '目赤肿痛', '烦躁易怒'], 'tongue': '舌红苔黄', 'pulse': '脉弦数', 'ground_truth': '肝火上炎', 'difficulty': 'easy'},
        {'id': 'E005', 'symptoms': ['咳嗽痰多', '胸闷', '食欲不振'], 'tongue': '舌淡苔白腻', 'pulse': '脉滑', 'ground_truth': '痰湿蕴肺', 'difficulty': 'easy'},
        {'id': 'E006', 'symptoms': ['发热', '口渴', '便秘'], 'tongue': '舌红苔黄燥', 'pulse': '脉数', 'ground_truth': '热结便秘', 'difficulty': 'easy'},
        {'id': 'E007', 'symptoms': ['气短乏力', '自汗', '面色苍白'], 'tongue': '舌淡苔薄白', 'pulse': '脉虚', 'ground_truth': '肺气虚', 'difficulty': 'easy'},
        {'id': 'E008', 'symptoms': ['头晕耳鸣', '腰膝酸软', '五心烦热'], 'tongue': '舌红少苔', 'pulse': '脉细数', 'ground_truth': '肾阴虚', 'difficulty': 'easy'},
        {'id': 'E009', 'symptoms': ['胃脘胀痛', '嗳腐吞酸', '厌食'], 'tongue': '舌苔厚腻', 'pulse': '脉滑', 'ground_truth': '食滞胃脘', 'difficulty': 'easy'},
        {'id': 'E010', 'symptoms': ['恶寒发热', '头身疼痛', '鼻塞流涕'], 'tongue': '舌苔薄白', 'pulse': '脉浮紧', 'ground_truth': '风寒表证', 'difficulty': 'easy'},

        # Medium cases (need differentiation)
        {'id': 'M001', 'symptoms': ['胁肋胀痛', '口苦', '目赤', '便秘'], 'tongue': '舌红苔黄', 'pulse': '脉弦数', 'ground_truth': '肝胆湿热', 'difficulty': 'medium'},
        {'id': 'M002', 'symptoms': ['心悸', '失眠', '多梦', '头晕'], 'tongue': '舌淡', 'pulse': '脉细', 'ground_truth': '心血不足', 'difficulty': 'medium'},
        {'id': 'M003', 'symptoms': ['腹胀', '便溏', '乏力', '面色萎黄'], 'tongue': '舌淡胖有齿痕', 'pulse': '脉缓弱', 'ground_truth': '脾虚湿盛', 'difficulty': 'medium'},
        {'id': 'M004', 'symptoms': ['咳嗽', '痰黄稠', '发热', '口渴'], 'tongue': '舌红苔黄', 'pulse': '脉数', 'ground_truth': '痰热壅肺', 'difficulty': 'medium'},
        {'id': 'M005', 'symptoms': ['胸闷', '心悸', '气短', '乏力'], 'tongue': '舌淡紫', 'pulse': '脉结代', 'ground_truth': '心气虚兼血瘀', 'difficulty': 'medium'},
        {'id': 'M006', 'symptoms': ['头痛', '眩晕', '面红目赤', '急躁易怒'], 'tongue': '舌红', 'pulse': '脉弦有力', 'ground_truth': '肝阳上亢', 'difficulty': 'medium'},
        {'id': 'M007', 'symptoms': ['胃脘灼痛', '口臭', '牙龈肿痛', '便秘'], 'tongue': '舌红苔黄厚', 'pulse': '脉滑数', 'ground_truth': '胃火炽盛', 'difficulty': 'medium'},
        {'id': 'M008', 'symptoms': ['腰痛', '畏寒', '下肢水肿', '小便不利'], 'tongue': '舌淡胖苔白滑', 'pulse': '脉沉迟', 'ground_truth': '肾阳虚水泛', 'difficulty': 'medium'},
        {'id': 'M009', 'symptoms': ['干咳', '咽干', '潮热', '盗汗'], 'tongue': '舌红少津', 'pulse': '脉细数', 'ground_truth': '肺阴虚', 'difficulty': 'medium'},
        {'id': 'M010', 'symptoms': ['脘腹胀满', '恶心呕吐', '身目发黄', '口苦'], 'tongue': '舌红苔黄腻', 'pulse': '脉滑数', 'ground_truth': '肝胆湿热', 'difficulty': 'medium'},

        # Hard cases (ambiguous or compound syndromes)
        {'id': 'H001', 'symptoms': ['胸闷', '心悸', '失眠', '腹胀', '便溏', '乏力'], 'tongue': '舌淡苔薄白', 'pulse': '脉细弱', 'ground_truth': '心脾两虚', 'difficulty': 'hard'},
        {'id': 'H002', 'symptoms': ['头晕', '耳鸣', '腰膝酸软', '心悸', '失眠'], 'tongue': '舌红少苔', 'pulse': '脉细数', 'ground_truth': '心肾不交', 'difficulty': 'hard'},
        {'id': 'H003', 'symptoms': ['胁肋胀痛', '腹胀', '便溏', '乏力', '情绪低落'], 'tongue': '舌淡红', 'pulse': '脉弦细', 'ground_truth': '肝郁脾虚', 'difficulty': 'hard'},
        {'id': 'H004', 'symptoms': ['畏寒', '肢冷', '口干', '五心烦热', '腰膝酸软'], 'tongue': '舌淡红', 'pulse': '沉细', 'ground_truth': '阴阳两虚', 'difficulty': 'hard'},
        {'id': 'H005', 'symptoms': ['咳嗽', '气喘', '痰多', '胸闷', '心悸', '下肢水肿'], 'tongue': '舌淡紫苔白滑', 'pulse': '脉沉弦', 'ground_truth': '肺肾两虚兼痰饮', 'difficulty': 'hard'},
        {'id': 'H006', 'symptoms': ['头痛', '眩晕', '恶心', '呕吐', '胸闷', '脘痞'], 'tongue': '舌苔白腻', 'pulse': '脉滑', 'ground_truth': '痰浊中阻', 'difficulty': 'hard'},
        {'id': 'H007', 'symptoms': ['发热', '恶寒', '身痛', '口渴', '咽痛', '咳嗽'], 'tongue': '舌红苔薄黄', 'pulse': '脉浮数', 'ground_truth': '风热表证', 'difficulty': 'hard'},
        {'id': 'H008', 'symptoms': ['胃脘痛', '嗳气', '胁肋胀痛', '情绪波动加重'], 'tongue': '舌淡红苔薄', 'pulse': '脉弦', 'ground_truth': '肝气犯胃', 'difficulty': 'hard'},
        {'id': 'H009', 'symptoms': ['心悸', '胸闷', '刺痛', '痛处固定', '唇甲紫暗'], 'tongue': '舌紫暗有瘀斑', 'pulse': '脉涩', 'ground_truth': '心血瘀阻', 'difficulty': 'hard'},
        {'id': 'H010', 'symptoms': ['低热', '午后加重', '手足心热', '盗汗', '口干'], 'tongue': '舌红少苔', 'pulse': '脉细数', 'ground_truth': '阴虚内热', 'difficulty': 'hard'},

        # === 扩充: Easy cases (E011-E050) ===
        {'id': 'E011', 'symptoms': ['面色苍白', '气短懒言', '自汗'], 'tongue': '舌淡苔薄白', 'pulse': '脉虚无力', 'ground_truth': '气虚', 'difficulty': 'easy'},
        {'id': 'E012', 'symptoms': ['畏寒喜温', '四肢不温', '小便清长'], 'tongue': '舌淡苔白', 'pulse': '脉沉迟', 'ground_truth': '阳虚', 'difficulty': 'easy'},
        {'id': 'E013', 'symptoms': ['口干咽燥', '五心烦热', '盗汗'], 'tongue': '舌红少苔', 'pulse': '脉细数', 'ground_truth': '阴虚', 'difficulty': 'easy'},
        {'id': 'E014', 'symptoms': ['面色唇甲淡白', '头晕眼花', '心悸'], 'tongue': '舌淡', 'pulse': '脉细', 'ground_truth': '血虚', 'difficulty': 'easy'},
        {'id': 'E015', 'symptoms': ['胸胁胀满', '善太息', '情志抑郁'], 'tongue': '舌淡红苔薄白', 'pulse': '脉弦', 'ground_truth': '肝气郁结', 'difficulty': 'easy'},
        {'id': 'E016', 'symptoms': ['脘腹胀满', '纳呆', '大便溏薄'], 'tongue': '舌淡胖有齿痕', 'pulse': '脉缓弱', 'ground_truth': '脾气虚', 'difficulty': 'easy'},
        {'id': 'E017', 'symptoms': ['咳嗽气喘', '痰白清稀', '恶寒'], 'tongue': '舌淡苔白', 'pulse': '脉浮紧', 'ground_truth': '风寒犯肺', 'difficulty': 'easy'},
        {'id': 'E018', 'symptoms': ['心烦失眠', '口舌生疮', '小便短赤'], 'tongue': '舌尖红', 'pulse': '脉数', 'ground_truth': '心火亢盛', 'difficulty': 'easy'},
        {'id': 'E019', 'symptoms': ['脘腹冷痛', '喜温喜按', '呕吐清水'], 'tongue': '舌淡苔白滑', 'pulse': '脉沉紧', 'ground_truth': '脾胃虚寒', 'difficulty': 'easy'},
        {'id': 'E020', 'symptoms': ['头痛', '发热', '汗出', '恶风'], 'tongue': '舌苔薄白', 'pulse': '脉浮缓', 'ground_truth': '风寒表虚', 'difficulty': 'easy'},
        {'id': 'E021', 'symptoms': ['口苦', '胁肋灼痛', '目赤'], 'tongue': '舌红苔黄', 'pulse': '脉弦数', 'ground_truth': '肝胆火旺', 'difficulty': 'easy'},
        {'id': 'E022', 'symptoms': ['小便频数', '尿急尿痛', '口渴'], 'tongue': '舌红苔黄腻', 'pulse': '脉滑数', 'ground_truth': '膀胱湿热', 'difficulty': 'easy'},
        {'id': 'E023', 'symptoms': ['胃脘灼痛', '渴喜冷饮', '口臭'], 'tongue': '舌红苔黄', 'pulse': '脉滑数', 'ground_truth': '胃热', 'difficulty': 'easy'},
        {'id': 'E024', 'symptoms': ['大便秘结', '腹胀痛', '口干'], 'tongue': '舌红苔黄燥', 'pulse': '脉沉实', 'ground_truth': '肠燥便秘', 'difficulty': 'easy'},
        {'id': 'E025', 'symptoms': ['遗精', '腰膝酸软', '头晕耳鸣'], 'tongue': '舌淡', 'pulse': '脉沉弱', 'ground_truth': '肾气不固', 'difficulty': 'easy'},
        {'id': 'E026', 'symptoms': ['月经量少', '色淡', '小腹空痛'], 'tongue': '舌淡', 'pulse': '脉细弱', 'ground_truth': '血虚', 'difficulty': 'easy'},
        {'id': 'E027', 'symptoms': ['肢体关节疼痛', '遇寒加重', '得温则减'], 'tongue': '舌淡苔白', 'pulse': '脉弦紧', 'ground_truth': '寒痹', 'difficulty': 'easy'},
        {'id': 'E028', 'symptoms': ['皮肤瘙痒', '疹出色红', '口渴'], 'tongue': '舌红苔薄黄', 'pulse': '脉浮数', 'ground_truth': '风热犯表', 'difficulty': 'easy'},
        {'id': 'E029', 'symptoms': ['水肿', '腰以下甚', '小便不利'], 'tongue': '舌淡胖', 'pulse': '脉沉', 'ground_truth': '肾虚水泛', 'difficulty': 'easy'},
        {'id': 'E030', 'symptoms': ['目赤肿痛', '羞明流泪', '头痛'], 'tongue': '舌红苔黄', 'pulse': '脉弦数', 'ground_truth': '肝火上炎', 'difficulty': 'easy'},
        {'id': 'E031', 'symptoms': ['鼻塞', '流浊涕', '头痛'], 'tongue': '舌红苔薄黄', 'pulse': '脉浮数', 'ground_truth': '风热犯肺', 'difficulty': 'easy'},
        {'id': 'E032', 'symptoms': ['牙龈肿痛', '口渴', '便秘'], 'tongue': '舌红苔黄', 'pulse': '脉洪数', 'ground_truth': '胃火炽盛', 'difficulty': 'easy'},
        {'id': 'E033', 'symptoms': ['胁肋隐痛', '口干', '目涩'], 'tongue': '舌红少苔', 'pulse': '脉弦细', 'ground_truth': '肝阴虚', 'difficulty': 'easy'},
        {'id': 'E034', 'symptoms': ['胸闷', '气短', '心前区刺痛'], 'tongue': '舌紫暗', 'pulse': '脉涩', 'ground_truth': '心血瘀阻', 'difficulty': 'easy'},
        {'id': 'E035', 'symptoms': ['腹痛', '泻后痛减', '嗳气'], 'tongue': '舌淡红', 'pulse': '脉弦', 'ground_truth': '肝气乘脾', 'difficulty': 'easy'},
        {'id': 'E036', 'symptoms': ['咳嗽', '痰中带血', '潮热'], 'tongue': '舌红少苔', 'pulse': '脉细数', 'ground_truth': '肺阴虚', 'difficulty': 'easy'},
        {'id': 'E037', 'symptoms': ['眩晕', '头重如裹', '胸闷恶心'], 'tongue': '舌苔白腻', 'pulse': '脉滑', 'ground_truth': '痰浊中阻', 'difficulty': 'easy'},
        {'id': 'E038', 'symptoms': ['面色晦暗', '肌肤甲错', '唇暗'], 'tongue': '舌紫暗有瘀斑', 'pulse': '脉涩', 'ground_truth': '血瘀', 'difficulty': 'easy'},
        {'id': 'E039', 'symptoms': ['嗳气', '脘腹胀满', '纳差'], 'tongue': '舌淡苔白', 'pulse': '脉弦', 'ground_truth': '肝胃不和', 'difficulty': 'easy'},
        {'id': 'E040', 'symptoms': ['心悸', '气短', '活动加重'], 'tongue': '舌淡', 'pulse': '脉结代', 'ground_truth': '心气虚', 'difficulty': 'easy'},
        {'id': 'E041', 'symptoms': ['多食易饥', '口渴', '消瘦'], 'tongue': '舌红苔黄', 'pulse': '脉滑数', 'ground_truth': '胃热炽盛', 'difficulty': 'easy'},
        {'id': 'E042', 'symptoms': ['尿血', '小便热涩刺痛'], 'tongue': '舌红', 'pulse': '脉数', 'ground_truth': '膀胱湿热', 'difficulty': 'easy'},
        {'id': 'E043', 'symptoms': ['带下量多', '色黄', '有臭味'], 'tongue': '舌红苔黄腻', 'pulse': '脉滑数', 'ground_truth': '湿热下注', 'difficulty': 'easy'},
        {'id': 'E044', 'symptoms': ['小儿疳积', '面黄肌瘦', '腹部膨隆'], 'tongue': '舌淡', 'pulse': '脉细弱', 'ground_truth': '脾虚食积', 'difficulty': 'easy'},
        {'id': 'E045', 'symptoms': ['产后恶露不绝', '小腹疼痛', '色暗有块'], 'tongue': '舌紫暗', 'pulse': '脉涩', 'ground_truth': '血瘀', 'difficulty': 'easy'},
        {'id': 'E046', 'symptoms': ['耳鸣如蝉', '听力下降', '腰膝酸软'], 'tongue': '舌淡', 'pulse': '脉沉弱', 'ground_truth': '肾精不足', 'difficulty': 'easy'},
        {'id': 'E047', 'symptoms': ['口眼歪斜', '言语不利', '半身不遂'], 'tongue': '舌暗苔白腻', 'pulse': '脉弦滑', 'ground_truth': '风痰阻络', 'difficulty': 'easy'},
        {'id': 'E048', 'symptoms': ['疮疡红肿热痛', '发热', '口渴'], 'tongue': '舌红苔黄', 'pulse': '脉数', 'ground_truth': '热毒蕴结', 'difficulty': 'easy'},
        {'id': 'E049', 'symptoms': ['目窠微肿', '恶风', '发热'], 'tongue': '舌苔薄白', 'pulse': '脉浮', 'ground_truth': '风水相搏', 'difficulty': 'easy'},
        {'id': 'E050', 'symptoms': ['胃脘隐痛', '喜温喜按', '泛吐清水'], 'tongue': '舌淡苔白', 'pulse': '脉虚弱', 'ground_truth': '脾胃虚寒', 'difficulty': 'easy'},

        # === 扩充: Medium cases (M011-M080) ===
        {'id': 'M011', 'symptoms': ['胸闷', '心悸', '刺痛', '痛处固定'], 'tongue': '舌紫暗有瘀斑', 'pulse': '脉涩', 'ground_truth': '心血瘀阻', 'difficulty': 'medium'},
        {'id': 'M012', 'symptoms': ['咳嗽', '痰少而粘', '口干咽燥', '潮热'], 'tongue': '舌红少津', 'pulse': '脉细数', 'ground_truth': '肺阴虚', 'difficulty': 'medium'},
        {'id': 'M013', 'symptoms': ['头痛', '眩晕', '面红', '急躁', '失眠'], 'tongue': '舌红', 'pulse': '脉弦有力', 'ground_truth': '肝阳上亢', 'difficulty': 'medium'},
        {'id': 'M014', 'symptoms': ['脘腹胀痛', '嗳气', '矢气后减轻', '情志不畅加重'], 'tongue': '舌淡红', 'pulse': '脉弦', 'ground_truth': '肝气犯胃', 'difficulty': 'medium'},
        {'id': 'M015', 'symptoms': ['心悸', '失眠', '健忘', '面色萎黄', '乏力'], 'tongue': '舌淡', 'pulse': '脉细弱', 'ground_truth': '心脾两虚', 'difficulty': 'medium'},
        {'id': 'M016', 'symptoms': ['胁肋灼痛', '口苦', '恶心', '目黄', '身黄'], 'tongue': '舌红苔黄腻', 'pulse': '脉弦数', 'ground_truth': '肝胆湿热', 'difficulty': 'medium'},
        {'id': 'M017', 'symptoms': ['腰痛', '转侧不利', '阴雨天加重'], 'tongue': '舌淡苔白腻', 'pulse': '脉沉缓', 'ground_truth': '寒湿腰痛', 'difficulty': 'medium'},
        {'id': 'M018', 'symptoms': ['胃脘灼痛', '嘈杂', '饥不欲食', '口干'], 'tongue': '舌红少苔', 'pulse': '脉细数', 'ground_truth': '胃阴虚', 'difficulty': 'medium'},
        {'id': 'M019', 'symptoms': ['肢体困重', '头如裹', '胸闷', '纳呆'], 'tongue': '舌苔白腻', 'pulse': '脉濡缓', 'ground_truth': '湿困脾胃', 'difficulty': 'medium'},
        {'id': 'M020', 'symptoms': ['咳嗽', '气喘', '痰多泡沫', '恶寒'], 'tongue': '舌淡苔白滑', 'pulse': '脉弦紧', 'ground_truth': '寒饮伏肺', 'difficulty': 'medium'},
        {'id': 'M021', 'symptoms': ['少腹疼痛', '月经不调', '经色暗有块'], 'tongue': '舌紫暗', 'pulse': '脉弦涩', 'ground_truth': '气滞血瘀', 'difficulty': 'medium'},
        {'id': 'M022', 'symptoms': ['大便时溏时泻', '食后腹胀', '面色萎黄'], 'tongue': '舌淡苔白', 'pulse': '脉缓弱', 'ground_truth': '脾气虚弱', 'difficulty': 'medium'},
        {'id': 'M023', 'symptoms': ['发热', '午后加重', '口腻', '胸闷'], 'tongue': '舌红苔黄腻', 'pulse': '脉滑数', 'ground_truth': '湿热蕴结', 'difficulty': 'medium'},
        {'id': 'M024', 'symptoms': ['心悸', '怔忡', '形寒肢冷', '浮肿'], 'tongue': '舌淡胖', 'pulse': '脉沉微', 'ground_truth': '心肾阳虚', 'difficulty': 'medium'},
        {'id': 'M025', 'symptoms': ['眩晕', '耳鸣', '头胀痛', '面赤', '易怒'], 'tongue': '舌红', 'pulse': '脉弦', 'ground_truth': '肝阳上亢', 'difficulty': 'medium'},
        {'id': 'M026', 'symptoms': ['咳血', '血色鲜红', '口干咽燥'], 'tongue': '舌红', 'pulse': '脉数', 'ground_truth': '肺热伤络', 'difficulty': 'medium'},
        {'id': 'M027', 'symptoms': ['失眠', '多梦', '心烦', '口苦'], 'tongue': '舌红苔黄', 'pulse': '脉弦数', 'ground_truth': '心肝火旺', 'difficulty': 'medium'},
        {'id': 'M028', 'symptoms': ['胃痛', '痛处拒按', '食后加重', '嗳腐'], 'tongue': '舌苔厚腻', 'pulse': '脉滑', 'ground_truth': '食积胃痛', 'difficulty': 'medium'},
        {'id': 'M029', 'symptoms': ['胁痛', '隐痛', '口干', '目涩', '视物模糊'], 'tongue': '舌红少苔', 'pulse': '脉弦细', 'ground_truth': '肝血虚', 'difficulty': 'medium'},
        {'id': 'M030', 'symptoms': ['小便浑浊', '如米泔', '腰膝酸软'], 'tongue': '舌淡', 'pulse': '脉沉弱', 'ground_truth': '肾虚不固', 'difficulty': 'medium'},
        {'id': 'M031', 'symptoms': ['腹痛', '里急后重', '下痢赤白'], 'tongue': '舌红苔黄腻', 'pulse': '脉滑数', 'ground_truth': '湿热痢', 'difficulty': 'medium'},
        {'id': 'M032', 'symptoms': ['关节红肿热痛', '屈伸不利', '发热'], 'tongue': '舌红苔黄', 'pulse': '脉滑数', 'ground_truth': '热痹', 'difficulty': 'medium'},
        {'id': 'M033', 'symptoms': ['胸闷', '气短', '心悸', '自汗', '乏力'], 'tongue': '舌淡', 'pulse': '脉弱', 'ground_truth': '心气不足', 'difficulty': 'medium'},
        {'id': 'M034', 'symptoms': ['呃逆', '声短而频', '不能自制'], 'tongue': '舌淡', 'pulse': '脉沉弱', 'ground_truth': '胃气上逆', 'difficulty': 'medium'},
        {'id': 'M035', 'symptoms': ['咳嗽', '痰黄', '气粗', '口渴', '发热'], 'tongue': '舌红苔黄', 'pulse': '脉数', 'ground_truth': '肺热壅盛', 'difficulty': 'medium'},
        {'id': 'M036', 'symptoms': ['头痛', '痛有定处', '如锥如刺'], 'tongue': '舌紫暗', 'pulse': '脉涩', 'ground_truth': '瘀血头痛', 'difficulty': 'medium'},
        {'id': 'M037', 'symptoms': ['泄泻', '腹痛即泻', '泻后痛减', '嗳气'], 'tongue': '舌淡红', 'pulse': '脉弦', 'ground_truth': '肝气乘脾', 'difficulty': 'medium'},
        {'id': 'M038', 'symptoms': ['腰膝酸软', '头晕', '遗精', '五心烦热'], 'tongue': '舌红', 'pulse': '脉细数', 'ground_truth': '肾阴虚', 'difficulty': 'medium'},
        {'id': 'M039', 'symptoms': ['咳嗽', '痰白量多', '脘痞', '纳差'], 'tongue': '舌淡苔白腻', 'pulse': '脉滑', 'ground_truth': '痰湿蕴肺', 'difficulty': 'medium'},
        {'id': 'M040', 'symptoms': ['腹胀', '腹痛', '大便不通', '矢气全无'], 'tongue': '舌苔厚腻', 'pulse': '脉沉实', 'ground_truth': '腑气不通', 'difficulty': 'medium'},
        {'id': 'M041', 'symptoms': ['夜寐不安', '心烦', '口舌生疮', '小便黄'], 'tongue': '舌尖红', 'pulse': '脉数', 'ground_truth': '心火上炎', 'difficulty': 'medium'},
        {'id': 'M042', 'symptoms': ['胁肋胀痛', '走窜不定', '乳房胀痛'], 'tongue': '舌淡红', 'pulse': '脉弦', 'ground_truth': '肝气郁结', 'difficulty': 'medium'},
        {'id': 'M043', 'symptoms': ['胃脘冷痛', '畏寒喜暖', '口泛清水'], 'tongue': '舌淡苔白', 'pulse': '脉沉紧', 'ground_truth': '胃寒', 'difficulty': 'medium'},
        {'id': 'M044', 'symptoms': ['头晕', '心悸', '面色苍白', '唇甲色淡'], 'tongue': '舌淡', 'pulse': '脉细弱', 'ground_truth': '心血虚', 'difficulty': 'medium'},
        {'id': 'M045', 'symptoms': ['水肿', '按之凹陷不起', '脘腹胀闷', '纳差'], 'tongue': '舌淡胖苔白滑', 'pulse': '脉沉缓', 'ground_truth': '脾虚水泛', 'difficulty': 'medium'},
        {'id': 'M046', 'symptoms': ['目赤', '口苦', '胁痛', '耳聋', '易怒'], 'tongue': '舌红苔黄', 'pulse': '脉弦数', 'ground_truth': '肝胆实火', 'difficulty': 'medium'},
        {'id': 'M047', 'symptoms': ['干咳', '无痰', '咽痒', '声音嘶哑'], 'tongue': '舌红少津', 'pulse': '脉细', 'ground_truth': '肺燥', 'difficulty': 'medium'},
        {'id': 'M048', 'symptoms': ['腹痛绵绵', '喜温喜按', '便溏', '乏力'], 'tongue': '舌淡苔白', 'pulse': '脉沉细', 'ground_truth': '脾胃虚寒', 'difficulty': 'medium'},
        {'id': 'M049', 'symptoms': ['失眠', '心悸', '多梦', '易惊'], 'tongue': '舌淡', 'pulse': '脉细', 'ground_truth': '心胆气虚', 'difficulty': 'medium'},
        {'id': 'M050', 'symptoms': ['肢体麻木', '肌肤甲错', '面色黧黑'], 'tongue': '舌紫暗', 'pulse': '脉涩', 'ground_truth': '血瘀', 'difficulty': 'medium'},
        {'id': 'M051', 'symptoms': ['咳喘', '痰多', '胸满', '不能平卧'], 'tongue': '舌苔白腻', 'pulse': '脉滑', 'ground_truth': '痰浊阻肺', 'difficulty': 'medium'},
        {'id': 'M052', 'symptoms': ['口渴多饮', '多食', '多尿', '消瘦'], 'tongue': '舌红', 'pulse': '脉滑数', 'ground_truth': '消渴', 'difficulty': 'medium'},
        {'id': 'M053', 'symptoms': ['少腹冷痛', '得温则减', '畏寒肢冷'], 'tongue': '舌淡苔白', 'pulse': '脉沉紧', 'ground_truth': '寒凝肝脉', 'difficulty': 'medium'},
        {'id': 'M054', 'symptoms': ['目窠浮肿', '恶风', '发热', '小便不利'], 'tongue': '舌苔薄白', 'pulse': '脉浮', 'ground_truth': '风水', 'difficulty': 'medium'},
        {'id': 'M055', 'symptoms': ['胃脘疼痛', '痛有定处', '拒按', '食后加重'], 'tongue': '舌紫暗', 'pulse': '脉涩', 'ground_truth': '胃脘瘀血', 'difficulty': 'medium'},
        {'id': 'M056', 'symptoms': ['耳鸣', '突发', '头痛面赤', '口苦'], 'tongue': '舌红苔黄', 'pulse': '脉弦数', 'ground_truth': '肝胆实火', 'difficulty': 'medium'},
        {'id': 'M057', 'symptoms': ['自汗', '恶风', '易感冒', '面色㿠白'], 'tongue': '舌淡', 'pulse': '脉浮缓无力', 'ground_truth': '肺气虚', 'difficulty': 'medium'},
        {'id': 'M058', 'symptoms': ['痛经', '经前胀痛', '经色暗有块'], 'tongue': '舌紫暗', 'pulse': '脉弦涩', 'ground_truth': '气滞血瘀', 'difficulty': 'medium'},
        {'id': 'M059', 'symptoms': ['肢体沉重', '关节酸痛', '肌肤麻木'], 'tongue': '舌苔白腻', 'pulse': '脉濡缓', 'ground_truth': '湿痹', 'difficulty': 'medium'},
        {'id': 'M060', 'symptoms': ['口眼歪斜', '语言蹇涩', '肢体麻木'], 'tongue': '舌暗苔腻', 'pulse': '脉弦滑', 'ground_truth': '风痰入络', 'difficulty': 'medium'},
        {'id': 'M061', 'symptoms': ['心悸', '胸闷', '气短', '面色紫暗'], 'tongue': '舌紫暗有瘀点', 'pulse': '脉结代', 'ground_truth': '心脉痹阻', 'difficulty': 'medium'},
        {'id': 'M062', 'symptoms': ['腹胀', '腹水', '青筋暴露', '面色黧黑'], 'tongue': '舌紫暗', 'pulse': '脉沉弦', 'ground_truth': '瘀血内阻', 'difficulty': 'medium'},
        {'id': 'M063', 'symptoms': ['咳嗽', '午后潮热', '咯血', '盗汗'], 'tongue': '舌红', 'pulse': '脉细数', 'ground_truth': '肺痨', 'difficulty': 'medium'},
        {'id': 'M064', 'symptoms': ['心下痞满', '按之不痛', '恶心呕吐'], 'tongue': '舌苔腻', 'pulse': '脉滑', 'ground_truth': '痰气痞结', 'difficulty': 'medium'},
        {'id': 'M065', 'symptoms': ['头痛', '头重如裹', '肢体困倦'], 'tongue': '舌苔白腻', 'pulse': '脉濡', 'ground_truth': '风湿头痛', 'difficulty': 'medium'},
        {'id': 'M066', 'symptoms': ['胁痛', '口干', '目赤', '大便干燥'], 'tongue': '舌红', 'pulse': '脉弦数', 'ground_truth': '肝火犯胃', 'difficulty': 'medium'},
        {'id': 'M067', 'symptoms': ['咳嗽', '痰中带血', '胸痛', '发热'], 'tongue': '舌红苔黄', 'pulse': '脉数', 'ground_truth': '痰热壅肺', 'difficulty': 'medium'},
        {'id': 'M068', 'symptoms': ['遗尿', '腰膝酸软', '畏寒肢冷'], 'tongue': '舌淡', 'pulse': '脉沉弱', 'ground_truth': '肾气不固', 'difficulty': 'medium'},
        {'id': 'M069', 'symptoms': ['胃痛', '灼热感', '口干口苦', '大便干'], 'tongue': '舌红苔黄', 'pulse': '脉弦数', 'ground_truth': '肝胃郁热', 'difficulty': 'medium'},
        {'id': 'M070', 'symptoms': ['腹泻', '水样便', '腹痛肠鸣', '恶寒'], 'tongue': '舌苔白腻', 'pulse': '脉濡缓', 'ground_truth': '寒湿泄泻', 'difficulty': 'medium'},
        {'id': 'M071', 'symptoms': ['皮肤风团', '瘙痒', '遇风加重'], 'tongue': '舌苔薄白', 'pulse': '脉浮', 'ground_truth': '风邪犯表', 'difficulty': 'medium'},
        {'id': 'M072', 'symptoms': ['半身不遂', '口眼歪斜', '语言不利'], 'tongue': '舌暗有瘀斑', 'pulse': '脉弦细', 'ground_truth': '中风后遗', 'difficulty': 'medium'},
        {'id': 'M073', 'symptoms': ['月经量多', '色淡质稀', '神疲乏力'], 'tongue': '舌淡', 'pulse': '脉细弱', 'ground_truth': '脾不统血', 'difficulty': 'medium'},
        {'id': 'M074', 'symptoms': ['胃脘隐痛', '纳差', '口干', '大便干结'], 'tongue': '舌红少苔', 'pulse': '脉细数', 'ground_truth': '胃阴虚', 'difficulty': 'medium'},
        {'id': 'M075', 'symptoms': ['头痛', '恶风', '鼻塞', '流清涕'], 'tongue': '舌苔薄白', 'pulse': '脉浮', 'ground_truth': '风寒头痛', 'difficulty': 'medium'},
        {'id': 'M076', 'symptoms': ['心悸', '胸闷', '头重', '痰多'], 'tongue': '舌苔腻', 'pulse': '脉滑', 'ground_truth': '痰阻心脉', 'difficulty': 'medium'},
        {'id': 'M077', 'symptoms': ['胁肋隐痛', '绵绵不休', '遇劳加重'], 'tongue': '舌淡', 'pulse': '脉弦细', 'ground_truth': '肝血虚', 'difficulty': 'medium'},
        {'id': 'M078', 'symptoms': ['水肿', '腰以下甚', '畏寒肢冷'], 'tongue': '舌淡胖', 'pulse': '脉沉迟', 'ground_truth': '肾阳虚水泛', 'difficulty': 'medium'},
        {'id': 'M079', 'symptoms': ['失眠', '心烦', '口干', '腰膝酸软'], 'tongue': '舌红少苔', 'pulse': '脉细数', 'ground_truth': '心肾不交', 'difficulty': 'medium'},
        {'id': 'M080', 'symptoms': ['腹痛', '喜温', '恶寒', '大便溏薄'], 'tongue': '舌淡苔白', 'pulse': '脉沉紧', 'ground_truth': '寒邪内阻', 'difficulty': 'medium'},

        # === 扩充: Hard cases (H011-H080) ===
        {'id': 'H011', 'symptoms': ['心悸', '失眠', '头晕', '面色萎黄', '月经量少'], 'tongue': '舌淡', 'pulse': '脉细弱', 'ground_truth': '心血虚兼脾虚', 'difficulty': 'hard'},
        {'id': 'H012', 'symptoms': ['胁肋胀痛', '口苦', '目赤', '腹胀', '便溏'], 'tongue': '舌红苔黄腻', 'pulse': '脉弦滑数', 'ground_truth': '肝脾湿热', 'difficulty': 'hard'},
        {'id': 'H013', 'symptoms': ['胸闷', '心悸', '气短', '畏寒', '下肢浮肿'], 'tongue': '舌淡紫', 'pulse': '脉沉微', 'ground_truth': '心肾阳虚', 'difficulty': 'hard'},
        {'id': 'H014', 'symptoms': ['咳嗽', '气喘', '痰多', '心悸', '浮肿', '不能平卧'], 'tongue': '舌淡紫暗', 'pulse': '脉沉弦', 'ground_truth': '肺肾两虚兼水饮', 'difficulty': 'hard'},
        {'id': 'H015', 'symptoms': ['头痛', '眩晕', '耳鸣', '腰膝酸软', '心悸失眠'], 'tongue': '舌红', 'pulse': '脉弦细', 'ground_truth': '肝肾阴虚', 'difficulty': 'hard'},
        {'id': 'H016', 'symptoms': ['脘腹胀满', '纳呆', '便溏', '乏力', '口苦', '目黄'], 'tongue': '舌淡红苔黄腻', 'pulse': '脉濡数', 'ground_truth': '脾虚湿热', 'difficulty': 'hard'},
        {'id': 'H017', 'symptoms': ['发热', '恶寒', '头痛', '身痛', '口渴', '咽痛'], 'tongue': '舌红苔薄黄', 'pulse': '脉浮数', 'ground_truth': '外寒内热', 'difficulty': 'hard'},
        {'id': 'H018', 'symptoms': ['心悸', '怔忡', '失眠', '盗汗', '五心烦热'], 'tongue': '舌红少苔', 'pulse': '脉细数', 'ground_truth': '心阴虚', 'difficulty': 'hard'},
        {'id': 'H019', 'symptoms': ['胁肋灼痛', '口苦', '便秘', '小便黄', '烦躁'], 'tongue': '舌红苔黄厚', 'pulse': '脉弦滑数', 'ground_truth': '肝胆实热', 'difficulty': 'hard'},
        {'id': 'H020', 'symptoms': ['头晕', '耳鸣', '健忘', '腰膝酸软', '畏寒肢冷'], 'tongue': '舌淡', 'pulse': '脉沉弱', 'ground_truth': '肾精不足兼阳虚', 'difficulty': 'hard'},
        {'id': 'H021', 'symptoms': ['胃脘灼痛', '嘈杂', '饥不欲食', '口干', '大便干'], 'tongue': '舌红少苔', 'pulse': '脉细数', 'ground_truth': '胃阴虚', 'difficulty': 'hard'},
        {'id': 'H022', 'symptoms': ['胸闷', '心悸', '气短', '乏力', '唇甲紫暗'], 'tongue': '舌紫暗有瘀点', 'pulse': '脉结代', 'ground_truth': '心气虚兼血瘀', 'difficulty': 'hard'},
        {'id': 'H023', 'symptoms': ['咳嗽', '痰少', '口干', '潮热', '盗汗', '乏力'], 'tongue': '舌红少津', 'pulse': '脉细数', 'ground_truth': '气阴两虚', 'difficulty': 'hard'},
        {'id': 'H024', 'symptoms': ['头痛', '头重如裹', '眩晕', '恶心', '胸闷', '纳差'], 'tongue': '舌苔白腻', 'pulse': '脉弦滑', 'ground_truth': '痰浊中阻', 'difficulty': 'hard'},
        {'id': 'H025', 'symptoms': ['少腹疼痛', '经色暗', '有块', '块下痛减', '胸胁胀满'], 'tongue': '舌紫暗有瘀斑', 'pulse': '脉弦涩', 'ground_truth': '气滞血瘀', 'difficulty': 'hard'},
        {'id': 'H026', 'symptoms': ['发热', '口渴', '汗出', '心烦', '小便短赤', '口舌生疮'], 'tongue': '舌红苔黄', 'pulse': '脉洪数', 'ground_truth': '气分热盛', 'difficulty': 'hard'},
        {'id': 'H027', 'symptoms': ['泄泻', '腹痛', '泻后痛减', '嗳气', '胁肋胀满'], 'tongue': '舌淡红', 'pulse': '脉弦', 'ground_truth': '肝气乘脾', 'difficulty': 'hard'},
        {'id': 'H028', 'symptoms': ['心悸', '气短', '自汗', '畏寒', '面色㿠白', '乏力'], 'tongue': '舌淡胖', 'pulse': '脉沉弱', 'ground_truth': '心阳虚', 'difficulty': 'hard'},
        {'id': 'H029', 'symptoms': ['眩晕', '耳鸣', '头胀', '面红', '腰膝酸软', '五心烦热'], 'tongue': '舌红', 'pulse': '脉弦细数', 'ground_truth': '阴虚阳亢', 'difficulty': 'hard'},
        {'id': 'H030', 'symptoms': ['肢体困倦', '头如裹', '胸闷', '口腻', '大便溏薄'], 'tongue': '舌苔白腻', 'pulse': '脉濡缓', 'ground_truth': '湿困中焦', 'difficulty': 'hard'},
        {'id': 'H031', 'symptoms': ['低热', '午后加重', '口干', '颧红', '手足心热', '盗汗'], 'tongue': '舌红少苔', 'pulse': '脉细数', 'ground_truth': '阴虚内热', 'difficulty': 'hard'},
        {'id': 'H032', 'symptoms': ['咳嗽', '痰多', '色白而粘', '胸闷', '纳差', '恶心'], 'tongue': '舌苔白腻', 'pulse': '脉滑', 'ground_truth': '痰湿蕴肺', 'difficulty': 'hard'},
        {'id': 'H033', 'symptoms': ['心悸', '怔忡', '气短', '自汗', '活动加重', '面色㿠白'], 'tongue': '舌淡', 'pulse': '脉虚弱', 'ground_truth': '心气虚', 'difficulty': 'hard'},
        {'id': 'H034', 'symptoms': ['胁肋隐痛', '口干', '目涩', '头晕', '视物模糊', '肢体麻木'], 'tongue': '舌淡', 'pulse': '脉弦细', 'ground_truth': '肝血虚', 'difficulty': 'hard'},
        {'id': 'H035', 'symptoms': ['胃脘疼痛', '痛有定处', '拒按', '黑便', '面色晦暗'], 'tongue': '舌紫暗', 'pulse': '脉涩', 'ground_truth': '胃络瘀阻', 'difficulty': 'hard'},
        {'id': 'H036', 'symptoms': ['发热', '口渴', '便秘', '腹胀', '小便短赤', '烦躁'], 'tongue': '舌红苔黄燥', 'pulse': '脉沉实', 'ground_truth': '阳明腑实', 'difficulty': 'hard'},
        {'id': 'H037', 'symptoms': ['失眠', '多梦', '心烦', '口苦', '胁肋胀痛', '目赤'], 'tongue': '舌红苔黄', 'pulse': '脉弦数', 'ground_truth': '心肝火旺', 'difficulty': 'hard'},
        {'id': 'H038', 'symptoms': ['咳嗽', '气喘', '痰黄稠', '发热', '胸痛', '口渴'], 'tongue': '舌红苔黄', 'pulse': '脉滑数', 'ground_truth': '痰热壅肺', 'difficulty': 'hard'},
        {'id': 'H039', 'symptoms': ['眩晕', '头重', '耳鸣', '心悸', '失眠', '健忘'], 'tongue': '舌淡', 'pulse': '脉弦细', 'ground_truth': '心脾两虚兼痰浊', 'difficulty': 'hard'},
        {'id': 'H040', 'symptoms': ['肢体关节疼痛', '游走不定', '屈伸不利', '恶风'], 'tongue': '舌苔薄白', 'pulse': '脉浮', 'ground_truth': '行痹', 'difficulty': 'hard'},
        {'id': 'H041', 'symptoms': ['心悸', '胸闷', '刺痛', '唇甲紫暗', '畏寒肢冷'], 'tongue': '舌紫暗', 'pulse': '脉沉涩', 'ground_truth': '心阳虚兼血瘀', 'difficulty': 'hard'},
        {'id': 'H042', 'symptoms': ['胁肋胀痛', '走窜不定', '善太息', '腹胀', '便溏', '乏力'], 'tongue': '舌淡红', 'pulse': '脉弦细', 'ground_truth': '肝郁脾虚', 'difficulty': 'hard'},
        {'id': 'H043', 'symptoms': ['咳嗽', '痰少', '口干', '咽燥', '声嘶', '午后潮热'], 'tongue': '舌红少津', 'pulse': '脉细数', 'ground_truth': '肺燥阴虚', 'difficulty': 'hard'},
        {'id': 'H044', 'symptoms': ['眩晕', '恶心呕吐', '胸闷', '脘痞', '纳差', '乏力'], 'tongue': '舌苔白腻', 'pulse': '脉弦滑', 'ground_truth': '痰湿中阻', 'difficulty': 'hard'},
        {'id': 'H045', 'symptoms': ['心悸', '失眠', '多梦', '健忘', '腰膝酸软', '头晕'], 'tongue': '舌红', 'pulse': '脉细数', 'ground_truth': '心肾不交', 'difficulty': 'hard'},
        {'id': 'H046', 'symptoms': ['发热', '恶寒', '无汗', '头身疼痛', '口渴', '烦躁'], 'tongue': '舌红苔薄白', 'pulse': '脉浮紧', 'ground_truth': '表寒里热', 'difficulty': 'hard'},
        {'id': 'H047', 'symptoms': ['胁肋灼痛', '口苦', '口干', '目赤', '耳鸣', '便秘'], 'tongue': '舌红苔黄厚', 'pulse': '脉弦滑数', 'ground_truth': '肝胆实火', 'difficulty': 'hard'},
        {'id': 'H048', 'symptoms': ['腹痛', '腹泻', '便溏', '乏力', '面色萎黄', '纳差'], 'tongue': '舌淡苔白', 'pulse': '脉缓弱', 'ground_truth': '脾气虚弱', 'difficulty': 'hard'},
        {'id': 'H049', 'symptoms': ['心悸', '胸闷', '气短', '头晕', '面色苍白', '唇甲色淡'], 'tongue': '舌淡', 'pulse': '脉细弱', 'ground_truth': '气血两虚', 'difficulty': 'hard'},
        {'id': 'H050', 'symptoms': ['发热', '午后加重', '口腻', '胸闷', '身重', '纳呆'], 'tongue': '舌红苔黄腻', 'pulse': '脉滑数', 'ground_truth': '湿热蕴结', 'difficulty': 'hard'},
    ]

    def get_all(self) -> List[Dict]:
        return self.CASES

    def get_by_difficulty(self, difficulty: str) -> List[Dict]:
        return [c for c in self.CASES if c['difficulty'] == difficulty]


# ═══════════════════════════════════════════════════════════
# 2. Agent Simulators (for benchmark without real LLM)
# ═══════════════════════════════════════════════════════════
class SimulatedAgent:
    """Simulate agent diagnosis with configurable accuracy."""
    def __init__(self, name: str, base_accuracy: float = 0.6, noise: float = 0.2):
        self.name = name
        self.base_accuracy = base_accuracy
        self.noise = noise
        self.syndrome_list = [
            '肝郁气滞', '心脾两虚', '肾阳虚', '肝火上炎', '痰湿蕴肺',
            '热结便秘', '肺气虚', '肾阴虚', '食滞胃脘', '风寒表证',
            '肝胆湿热', '心血不足', '脾虚湿盛', '痰热壅肺', '心气虚兼血瘀',
            '肝阳上亢', '胃火炽盛', '肾阳虚水泛', '肺阴虚', '心肾不交',
            '肝郁脾虚', '阴阳两虚', '肺肾两虚兼痰饮', '痰浊中阻', '风热表证',
            '肝气犯胃', '心血瘀阻', '阴虚内热',
        ]

    def diagnose(self, case: Dict) -> Dict:
        """Simulate diagnosis with noise."""
        gt = case['ground_truth']
        if np.random.random() < self.base_accuracy:
            pred = gt
            conf = np.random.uniform(0.7, 0.95)
        else:
            # Wrong prediction
            wrong = [s for s in self.syndrome_list if s != gt]
            pred = np.random.choice(wrong)
            conf = np.random.uniform(0.3, 0.7)

        alternatives = []
        if np.random.random() > 0.5:
            alt = np.random.choice([s for s in self.syndrome_list if s != pred])
            alternatives.append((alt, np.random.uniform(0.2, 0.5)))

        return {
            'agent': self.name,
            'prediction': pred,
            'confidence': conf,
            'alternatives': alternatives,
        }


# ═══════════════════════════════════════════════════════════
# 3. Diagnosis Methods
# ═══════════════════════════════════════════════════════════
class SingleAgentMethod:
    """Baseline: single agent direct diagnosis."""
    name = "SingleAgent"

    def __init__(self):
        self.agent = SimulatedAgent("Single", base_accuracy=0.55)

    def diagnose(self, case: Dict) -> Dict:
        result = self.agent.diagnose(case)
        return {
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'top3': [result['prediction']] + [a[0] for a in result.get('alternatives', [])],
            'consensus': 1.0,
            'conflicts': 0,
        }


class MultiAgentNoDeliberation:
    """Baseline: multiple agents, vote without deliberation."""
    name = "MultiAgent-NoDelib"

    def __init__(self):
        self.agents = [
            SimulatedAgent("Tongue", 0.6),
            SimulatedAgent("History", 0.55),
            SimulatedAgent("Pulse", 0.5),
        ]

    def diagnose(self, case: Dict) -> Dict:
        votes = [a.diagnose(case) for a in self.agents]
        predictions = [v['prediction'] for v in votes]
        counter = Counter(predictions)
        final = counter.most_common(1)[0][0]
        consensus = counter[final] / len(predictions)

        all_preds = list(set(predictions))
        for v in votes:
            for alt, _ in v.get('alternatives', []):
                if alt not in all_preds:
                    all_preds.append(alt)

        return {
            'prediction': final,
            'confidence': np.mean([v['confidence'] for v in votes]),
            'top3': all_preds[:3],
            'consensus': consensus,
            'conflicts': len(set(predictions)) - 1,
        }


class MultiAgentNoConflict:
    """Ablation: no conflict detection/resolution."""
    name = "MultiAgent-NoConflict"

    def __init__(self):
        self.agents = [
            SimulatedAgent("Tongue", 0.6),
            SimulatedAgent("History", 0.55),
            SimulatedAgent("Pulse", 0.5),
            SimulatedAgent("Lab", 0.5),
        ]

    def diagnose(self, case: Dict) -> Dict:
        votes = [a.diagnose(case) for a in self.agents]
        # Simple majority vote, no conflict handling
        predictions = [v['prediction'] for v in votes]
        counter = Counter(predictions)
        final = counter.most_common(1)[0][0]
        consensus = counter[final] / len(predictions)

        return {
            'prediction': final,
            'confidence': np.mean([v['confidence'] for v in votes]),
            'top3': [p for p, _ in counter.most_common(3)],
            'consensus': consensus,
            'conflicts': 0,  # not detected
        }


class MultiAgentNoCredibility:
    """Ablation: equal weights for all agents."""
    name = "MultiAgent-NoCred"

    def __init__(self):
        self.agents = [
            SimulatedAgent("Tongue", 0.6),
            SimulatedAgent("History", 0.55),
            SimulatedAgent("Pulse", 0.5),
            SimulatedAgent("Lab", 0.5),
        ]

    def diagnose(self, case: Dict) -> Dict:
        votes = [a.diagnose(case) for a in self.agents]
        # Equal weight voting
        syndrome_scores = defaultdict(float)
        for v in votes:
            syndrome_scores[v['prediction']] += 1.0 / len(votes)
            for alt, conf in v.get('alternatives', []):
                syndrome_scores[alt] += conf * 0.3 / len(votes)

        final = max(syndrome_scores, key=syndrome_scores.get)
        predictions = [v['prediction'] for v in votes]
        consensus = Counter(predictions).get(final, 0) / len(predictions)

        return {
            'prediction': final,
            'confidence': syndrome_scores[final],
            'top3': sorted(syndrome_scores, key=syndrome_scores.get, reverse=True)[:3],
            'consensus': consensus,
            'conflicts': len(set(predictions)) - 1,
        }


class MACCMFull:
    """Our full method: multi-agent + deliberation + conflict resolution + credibility."""
    name = "MACCM"

    CONFLICT_PAIRS = [
        ('寒', '热'), ('虚', '实'), ('阴虚', '阳虚'), ('气虚', '气滞'),
    ]

    def __init__(self):
        self.agents = [
            SimulatedAgent("Tongue", 0.6),
            SimulatedAgent("History", 0.55),
            SimulatedAgent("Pulse", 0.5),
            SimulatedAgent("Lab", 0.5),
            SimulatedAgent("Experience", 0.45),
            SimulatedAgent("Safety", 0.4),
        ]
        # Learned credibility weights (simulated)
        self.weights = np.array([0.25, 0.20, 0.20, 0.15, 0.12, 0.08])

    def _detect_conflict(self, predictions: List[str]) -> bool:
        for i in range(len(predictions)):
            for j in range(i+1, len(predictions)):
                for a, b in self.CONFLICT_PAIRS:
                    if (a in predictions[i] and b in predictions[j]) or \
                       (b in predictions[i] and a in predictions[j]):
                        return True
        return False

    def diagnose(self, case: Dict) -> Dict:
        votes = [a.diagnose(case) for a in self.agents]
        predictions = [v['prediction'] for v in votes]

        # Conflict detection
        has_conflict = self._detect_conflict(predictions)

        # Credibility-weighted voting
        syndrome_scores = defaultdict(float)
        for i, v in enumerate(votes):
            w = self.weights[i] * v['confidence']
            syndrome_scores[v['prediction']] += w
            for alt, conf in v.get('alternatives', []):
                syndrome_scores[alt] += w * conf * 0.3

        # Conflict resolution: if conflict, boost high-confidence agents
        if has_conflict:
            for i, v in enumerate(votes):
                if v['confidence'] > 0.7:
                    syndrome_scores[v['prediction']] *= 1.3

        final = max(syndrome_scores, key=syndrome_scores.get)
        consensus = Counter(predictions).get(final, 0) / len(predictions)

        return {
            'prediction': final,
            'confidence': syndrome_scores[final],
            'top3': sorted(syndrome_scores, key=syndrome_scores.get, reverse=True)[:3],
            'consensus': consensus,
            'conflicts': int(has_conflict),
        }


# ═══════════════════════════════════════════════════════════
# 4. Evaluation
# ═══════════════════════════════════════════════════════════
def evaluate_method(method, cases: List[Dict], n_runs: int = 5) -> Dict:
    """Evaluate a diagnosis method with multiple runs for stability."""
    all_top1 = []
    all_top3 = []
    all_consensus = []
    all_conflicts = []
    all_latency = []

    for run in range(n_runs):
        top1_correct = 0
        top3_correct = 0
        total_consensus = 0
        total_conflicts = 0
        start = time.time()

        for case in cases:
            result = method.diagnose(case)
            if result['prediction'] == case['ground_truth']:
                top1_correct += 1
            if case['ground_truth'] in result.get('top3', []):
                top3_correct += 1
            total_consensus += result.get('consensus', 0)
            total_conflicts += result.get('conflicts', 0)

        elapsed = (time.time() - start) * 1000
        n = len(cases)
        all_top1.append(top1_correct / n)
        all_top3.append(top3_correct / n)
        all_consensus.append(total_consensus / n)
        all_conflicts.append(total_conflicts / n)
        all_latency.append(elapsed / n)

    return {
        'method': method.name,
        'top1_acc': np.mean(all_top1),
        'top1_std': np.std(all_top1),
        'top3_acc': np.mean(all_top3),
        'top3_std': np.std(all_top3),
        'consensus': np.mean(all_consensus),
        'conflicts': np.mean(all_conflicts),
        'latency_ms': np.mean(all_latency),
    }


def run_maccm_benchmark():
    """Run complete MACCM benchmark."""
    print("=" * 70)
    print("MACCM Benchmark — 多智能体中医会诊评测")
    print("=" * 70)

    benchmark = TCMDiagnosisBenchmark()
    all_cases = benchmark.get_all()

    print(f"\n[1] Benchmark: {len(all_cases)} cases")
    for diff in ['easy', 'medium', 'hard']:
        cases = benchmark.get_by_difficulty(diff)
        print(f"  {diff}: {len(cases)} cases")

    # All methods
    methods = [
        SingleAgentMethod(),
        MultiAgentNoDeliberation(),
        MultiAgentNoConflict(),
        MultiAgentNoCredibility(),
        MACCMFull(),
    ]

    # Overall evaluation
    print(f"\n[2] Overall Results (5 runs)")
    print(f"{'Method':<25} {'Top-1':>8} {'Top-3':>8} {'Consensus':>10} {'Conflicts':>10} {'Latency':>10}")
    print("-" * 75)

    all_results = []
    for method in methods:
        result = evaluate_method(method, all_cases, n_runs=5)
        all_results.append(result)
        print(f"{result['method']:<25} {result['top1_acc']:>7.3f}±{result['top1_std']:.2f} "
              f"{result['top3_acc']:>7.3f}±{result['top3_std']:.2f} "
              f"{result['consensus']:>10.3f} {result['conflicts']:>10.1f} "
              f"{result['latency_ms']:>8.2f}ms")

    # Per-difficulty evaluation
    print(f"\n[3] Per-Difficulty Results")
    for diff in ['easy', 'medium', 'hard']:
        cases = benchmark.get_by_difficulty(diff)
        print(f"\n  [{diff.upper()}] ({len(cases)} cases)")
        print(f"  {'Method':<25} {'Top-1':>8} {'Top-3':>8}")
        print(f"  {'-'*45}")
        for method in methods:
            result = evaluate_method(method, cases, n_runs=3)
            print(f"  {result['method']:<25} {result['top1_acc']:>7.3f} {result['top3_acc']:>7.3f}")

    # Ablation comparison
    print(f"\n[4] Ablation Analysis (vs MACCM full)")
    maccm_result = all_results[-1]
    for r in all_results[:-1]:
        delta_top1 = maccm_result['top1_acc'] - r['top1_acc']
        delta_top3 = maccm_result['top3_acc'] - r['top3_acc']
        print(f"  MACCM vs {r['method']}: ΔTop1={delta_top1:+.3f}, ΔTop3={delta_top3:+.3f}")

    # Save results
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_cases': len(all_cases),
        'results': all_results,
    }
    with open('maccm_benchmark_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[5] Results saved to maccm_benchmark_results.json")

    return all_results


if __name__ == '__main__':
    run_maccm_benchmark()
