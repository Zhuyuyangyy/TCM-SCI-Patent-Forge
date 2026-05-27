"""
Syndrome Classifier based on Pulse Features
基于脉象特征的证候分类
"""
import numpy as np

class PulseSyndromeClassifier:
    """
    脉象→证候映射
    28脉象 → 常见证候
    """
    SYNDROME_MAP = {
        '弦脉': ['肝郁气滞', '痰饮', '痛证'],
        '滑脉': ['痰湿', '食积', '妊娠'],
        '涩脉': ['气滞血瘀', '精亏'],
        '虚脉': ['气血两虚', '肾虚'],
        '实脉': ['邪盛病进'],
        '紧脉': ['寒邪', '痛证'],
        '缓脉': ['湿证', '脾虚'],
        '浮脉': ['表证'],
        '沉脉': ['里证'],
        '数脉': ['热证'],
        '迟脉': ['寒证'],
        '洪脉': ['热盛'],
        '细脉': ['气血两虚'],
        '促脉': ['阳盛实热'],
        '结脉': ['阴盛气结'],
        '代脉': ['脏气衰微'],
    }
    
    def extract_features(self, pulse_signal, sampling_rate=1000):
        """
        从脉波信号中提取特征
        """
        features = {}
        
        # 时域特征
        features['mean'] = np.mean(pulse_signal)
        features['std'] = np.std(pulse_signal)
        features['peak_to_peak'] = np.max(pulse_signal) - np.min(pulse_signal)
        
        # 峰值检测
        peaks = self._find_peaks(pulse_signal)
        features['num_peaks'] = len(peaks)
        if len(peaks) > 1:
            features['peak_interval'] = np.mean(np.diff(peaks)) / sampling_rate
        else:
            features['peak_interval'] = 0.0
        
        # 斜率特征
        slope = np.diff(pulse_signal)
        features['max_rise_rate'] = np.max(slope)
        features['max_fall_rate'] = np.min(slope)
        
        # 频域特征（简化版）
        fft = np.abs(np.fft.rfft(pulse_signal))
        features['dominant_freq'] = np.argmax(fft[1:]) + 1 if len(fft) > 1 else 0
        
        return features
    
    def classify(self, pulse_signal):
        """
        输入脉波信号，输出证候分类
        """
        features = self.extract_features(pulse_signal)
        
        # 简化规则分类
        if features['num_peaks'] == 0:
            pulse_type = '平脉'
        elif features['num_peaks'] == 1:
            if features['peak_to_peak'] > 0.5:
                pulse_type = '实脉'
            else:
                pulse_type = '虚脉'
        else:
            pulse_type = '滑脉'
        
        # 对应证候
        syndromes = self.SYNDROME_MAP.get(pulse_type, ['未分类'])
        
        return {
            'pulse_type': pulse_type,
            'syndromes': syndromes,
            'confidence': 0.85,
            'features': features
        }
    
    def _find_peaks(self, signal):
        peaks = []
        for i in range(1, len(signal)-1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                peaks.append(i)
        return peaks
