"""
声纹特征提取模块 - Voice Feature Extraction
基于AudioSense-TCM的纯NumPy实现，提取F0/共振峰/LPC特征，输出8维特征向量
"""

import numpy as np


class VoiceAnalyzer:
    """声纹分析器 - 从音频信号中提取8维特征"""
    
    # 采样率
    SAMPLE_RATE = 16000
    
    # 体质相关的声纹特征先验知识
    CONSTITUTION_VOICE_FEATURES = {
        '平和': {
            'f0': 180,        # 正常基频
            'f1': 500,        # 第一共振峰
            'f2': 1500,       # 第二共振峰
            'f3': 2500,       # 第三共振峰
            'lpc_1': 0.8,     # LPC系数1
            'lpc_2': -0.3,
            'jitter': 0.01,   # 基频抖动
            'shimmer': 0.03   # 振幅抖动
        },
        '气虚': {
            'f0': 140,        # 基频偏低
            'f1': 450,
            'f2': 1400,
            'f3': 2400,
            'lpc_1': 0.6,
            'lpc_2': -0.2,
            'jitter': 0.02,
            'shimmer': 0.05
        },
        '阳虚': {
            'f0': 120,        # 基频明显偏低
            'f1': 420,
            'f2': 1350,
            'f3': 2300,
            'lpc_1': 0.5,
            'lpc_2': -0.15,
            'jitter': 0.015,
            'shimmer': 0.04
        },
        '阴虚': {
            'f0': 220,        # 基频偏高
            'f1': 550,
            'f2': 1600,
            'f3': 2700,
            'lpc_1': 0.9,
            'lpc_2': -0.4,
            'jitter': 0.025,
            'shimmer': 0.06
        },
        '痰湿': {
            'f0': 150,        # 基频偏低，浊
            'f1': 480,
            'f2': 1450,
            'f3': 2450,
            'lpc_1': 0.7,
            'lpc_2': -0.25,
            'jitter': 0.012,
            'shimmer': 0.035
        },
        '湿热': {
            'f0': 190,        # 基频稍高
            'f1': 520,
            'f2': 1550,
            'f3': 2600,
            'lpc_1': 0.85,
            'lpc_2': -0.35,
            'jitter': 0.018,
            'shimmer': 0.045
        },
        '血瘀': {
            'f0': 160,        # 基频偏低，嘶哑
            'f1': 460,
            'f2': 1420,
            'f3': 2350,
            'lpc_1': 0.55,
            'lpc_2': -0.2,
            'jitter': 0.03,   # 抖动较大
            'shimmer': 0.07
        },
        '气郁': {
            'f0': 200,        # 基频偏高，闷
            'f1': 540,
            'f2': 1580,
            'f3': 2650,
            'lpc_1': 0.75,
            'lpc_2': -0.28,
            'jitter': 0.015,
            'shimmer': 0.04
        },
        '特禀': {
            'f0': 175,
            'f1': 490,
            'f2': 1480,
            'f3': 2480,
            'lpc_1': 0.72,
            'lpc_2': -0.26,
            'jitter': 0.02,
            'shimmer': 0.05
        }
    }
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.feature_dim = 8
        
    def extract_from_audio(self, audio_data):
        """
        从音频数据中提取声纹特征
        audio_data: numpy数组，音频波形数据
        
        返回: 8维特征向量 [f0, f1, f2, f3, lpc_1, lpc_2, jitter, shimmer]
        """
        if len(audio_data) < 256:
            # 数据太短，返回默认值
            return np.array([180, 500, 1500, 2500, 0.8, -0.3, 0.01, 0.03])
        
        # 预加重
        audio_data = self._preemphasis(audio_data)
        
        # 提取各特征
        f0 = self._extract_f0(audio_data)
        formants = self._extract_formants(audio_data)
        lpc_coeffs = self._extract_lpc(audio_data)
        jitter_val = self._extract_jitter(audio_data)
        shimmer_val = self._extract_shimmer(audio_data)
        
        # 组合8维特征
        features = np.array([
            f0,
            formants[0] if len(formants) > 0 else 500,
            formants[1] if len(formants) > 1 else 1500,
            formants[2] if len(formants) > 2 else 2500,
            lpc_coeffs[0] if len(lpc_coeffs) > 0 else 0.8,
            lpc_coeffs[1] if len(lpc_coeffs) > 1 else -0.3,
            jitter_val,
            shimmer_val
        ])
        
        return features
    
    def _preemphasis(self, audio):
        """预加重滤波"""
        preemphasis = 0.97
        return np.append(audio[0], audio[1:] - preemphasis * audio[:-1])
    
    def _extract_f0(self, audio):
        """
        提取基频F0 - 使用自相关法
        """
        # 分帧
        frame_length = 1024
        hop_length = 512
        
        # 计算自相关
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # 找峰值对应的周期
        min_period = int(self.sample_rate / 500)  # F0 max = 500Hz
        max_period = int(self.sample_rate / 50)    # F0 min = 50Hz
        
        if len(autocorr) <= max_period:
            return 180.0
        
        segment = autocorr[min_period:max_period]
        if len(segment) == 0:
            return 180.0
            
        peak_idx = np.argmax(segment)
        period = peak_idx + min_period
        
        f0 = self.sample_rate / period if period > 0 else 180.0
        return np.clip(f0, 50, 500)
    
    def _extract_formants(self, audio):
        """
        提取共振峰 - 使用LPC法
        返回前3个共振峰频率
        """
        # LPC阶数
        lpc_order = 12
        
        # 分帧
        frame_length = 1024
        hop_length = 512
        
        # 获取语音段
        if len(audio) < frame_length:
            return [500, 1500, 2500]
        
        # 使用汉宁窗
        window = np.hanning(frame_length)
        frame = audio[:frame_length] * window
        
        # LPC分析
        lpc_coeffs = self._lpc_analysis(frame, lpc_order)
        
        # 求LPC根得到共振峰
        roots = np.roots(lpc_coeffs)
        roots = [r for r in roots if np.imag(r) > 0]  # 只取上半平面
        
        # 计算频率
        formant_freqs = []
        for r in roots:
            freq = np.arctan2(np.imag(r), np.real(r)) * self.sample_rate / (2 * np.pi)
            if 50 < freq < 4000:
                formant_freqs.append(freq)
        
        formant_freqs.sort()
        return formant_freqs[:3]
    
    def _lpc_analysis(self, frame, order):
        """
        Levinson-Durbin算法进行LPC分析
        """
        n = len(frame)
        
        # 计算自相关
        r = np.zeros(order + 1)
        for i in range(order + 1):
            r[i] = np.dot(frame[:n-i], frame[i:])
        
        if r[0] == 0:
            return np.ones(order + 1)
        
        # Levinson-Durbin递归
        a = np.zeros(order + 1)
        a[0] = 1.0
        e = r[0]
        
        for i in range(1, order + 1):
            if e == 0:
                e = 1e-10
            acc = sum(a[k] * r[i-k] for k in range(1, i))
            if abs(e) < 1e-10:
                e = 1e-10
            lpc = (r[i] - acc) / e
            if abs(lpc) > 1:
                lpc = np.sign(lpc) * 0.99
            
            a_temp = a[1:i].copy()
            for j in range(1, i):
                a[j] -= lpc * a[i-j]
            
            e = (1 - lpc * lpc) * e
            if e < 1e-10:
                e = 1e-10
        
        return a
    
    def _extract_lpc(self, audio):
        """
        提取LPC系数特征
        """
        if len(audio) < 1024:
            return [0.8, -0.3]
        
        frame = audio[:1024] * np.hanning(1024)
        lpc_coeffs = self._lpc_analysis(frame, 8)
        
        # 返回前两个反射系数（与LPC系数相关）
        return [lpc_coeffs[1] if len(lpc_coeffs) > 1 else 0.8,
                lpc_coeffs[2] if len(lpc_coeffs) > 2 else -0.3]
    
    def _extract_jitter(self, audio):
        """
        提取基频抖动 - 反映音调稳定性
        """
        # 简化的抖动计算：相邻周期的差异
        frame_length = 512
        hop_length = 256
        
        periods = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i+frame_length]
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            min_period = int(self.sample_rate / 500)
            max_period = int(self.sample_rate / 50)
            
            if len(autocorr) > max_period:
                segment = autocorr[min_period:max_period]
                if len(segment) > 0:
                    peak_idx = np.argmax(segment)
                    period = peak_idx + min_period
                    if period > 0:
                        periods.append(self.sample_rate / period)
        
        if len(periods) < 2:
            return 0.015
        
        # 计算抖动：相邻F0差值的平均值
        jitter = np.mean(np.abs(np.diff(periods))) / np.mean(periods)
        return np.clip(jitter, 0.001, 0.1)
    
    def _extract_shimmer(self, audio):
        """
        提取振幅抖动 - 反映声音强度稳定性
        """
        frame_length = 512
        hop_length = 256
        
        amplitudes = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i+frame_length]
            rms = np.sqrt(np.mean(frame**2))
            amplitudes.append(rms)
        
        if len(amplitudes) < 2:
            return 0.04
        
        # 计算 shimmer：相邻振幅差值的平均值
        shimmer = np.mean(np.abs(np.diff(amplitudes))) / np.mean(amplitudes)
        return np.clip(shimmer, 0.001, 0.2)
    
    def extract_from_dict(self, voice_info):
        """
        从声纹信息字典中提取特征（用于模拟数据）
        voice_info: 包含f0, f1, f2, f3, lpc_1, lpc_2, jitter, shimmer
        """
        features = []
        features.append(voice_info.get('f0', 180))
        features.append(voice_info.get('f1', 500))
        features.append(voice_info.get('f2', 1500))
        features.append(voice_info.get('f3', 2500))
        features.append(voice_info.get('lpc_1', 0.8))
        features.append(voice_info.get('lpc_2', -0.3))
        features.append(voice_info.get('jitter', 0.01))
        features.append(voice_info.get('shimmer', 0.03))
        
        return np.array(features)
    
    def analyze(self, voice_data):
        """
        主分析函数：提取声纹特征
        """
        if isinstance(voice_data, dict):
            return self.extract_from_dict(voice_data)
        elif isinstance(voice_data, np.ndarray):
            return self.extract_from_audio(voice_data)
        else:
            raise ValueError("Unsupported voice data format")
    
    def get_constitution_prediction(self, features):
        """
        基于声纹特征预测体质倾向
        """
        predictions = {}
        features = np.array(features)
        
        for constitution, voice_feat in self.CONSTITUTION_VOICE_FEATURES.items():
            ref_features = self.extract_from_dict(voice_feat)
            similarity = self._cosine_similarity(features, ref_features)
            predictions[constitution] = max(0.0, min(1.0, similarity))
        
        # 归一化
        total = sum(predictions.values())
        if total > 0:
            predictions = {k: v/total for k, v in predictions.items()}
        
        return predictions
    
    def _cosine_similarity(self, a, b):
        """计算余弦相似度"""
        a = np.array(a).flatten()
        b = np.array(b).flatten()
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.5
        return 0.5 + 0.5 * (dot_product / (norm_a * norm_b))


def create_mock_voice_data(constitution=None):
    """
    创建模拟声纹数据用于测试
    """
    analyzer = VoiceAnalyzer()
    if constitution and constitution in analyzer.CONSTITUTION_VOICE_FEATURES:
        return analyzer.CONSTITUTION_VOICE_FEATURES[constitution]
    else:
        return {
            'f0': 180,
            'f1': 500,
            'f2': 1500,
            'f3': 2500,
            'lpc_1': 0.8,
            'lpc_2': -0.3,
            'jitter': 0.01,
            'shimmer': 0.03
        }


def generate_mock_audio(f0=180, duration=1.0, sample_rate=16000):
    """
    生成模拟音频信号（用于测试）
    f0: 基频
    duration: 时长（秒）
    sample_rate: 采样率
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 合成信号：基频 + 谐波
    audio = np.sin(2 * np.pi * f0 * t)
    audio += 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
    audio += 0.3 * np.sin(2 * np.pi * 3 * f0 * t)
    audio += 0.2 * np.sin(2 * np.pi * 4 * f0 * t)
    
    # 添加一些噪声
    audio += 0.05 * np.random.randn(len(audio))
    
    # 音量包络
    envelope = np.ones_like(audio)
    attack = int(0.05 * sample_rate)
    release = int(0.1 * sample_rate)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)
    
    return audio * envelope


if __name__ == "__main__":
    # 测试声纹分析器
    print("=" * 60)
    print("声纹分析器测试")
    print("=" * 60)
    
    analyzer = VoiceAnalyzer()
    
    # 测试不同体质的声纹
    test_constitutions = ['气虚', '阴虚', '湿热', '血瘀']
    
    for constitution in test_constitutions:
        voice_data = create_mock_voice_data(constitution)
        features = analyzer.analyze(voice_data)
        predictions = analyzer.get_constitution_prediction(features)
        
        print(f"\n【{constitution}】声纹特征:")
        print(f"  F0={features[0]:.1f}Hz, F1={features[1]:.1f}Hz, "
              f"F2={features[2]:.1f}Hz, F3={features[3]:.1f}Hz")
        print(f"  预测结果: {max(predictions, key=predictions.get)} "
              f"(置信度: {max(predictions.values()):.3f})")
