"""Audio Feature Extractor for TCM Hearing Diagnosis（纯NumPy）"""
import numpy as np

class AudioFeatureExtractor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def extract_fundamental_frequency(self, audio):
        # 自相关法提取基频F0
        audio = audio - np.mean(audio)
        n = len(audio)
        corr = np.correlate(audio[:min(n,2000)], audio[:min(n,2000)], mode='full')
        corr = corr[n-1:]
        min_period = int(self.sample_rate / 500)
        max_period = int(self.sample_rate / 50)
        if max_period >= len(corr):
            max_period = len(corr) - 1
        peaks = []
        for i in range(min_period+1, max_period):
            if corr[i] > corr[i-1] and corr[i] > corr[i+1]:
                peaks.append((corr[i], i))
        if peaks:
            f0 = self.sample_rate / max(peaks, key=lambda x: x[0])[1]
            return f0
        return 0.0
    
    def extract_formants(self, audio):
        # 简化LPC共振峰提取
        n = min(len(audio), 512)
        autocorr = np.correlate(audio[:n], audio[:n], mode='full')
        autocorr = autocorr[n-1:]
        order = 8
        if len(autocorr) <= order:
            return [500.0, 1500.0, 2500.0]
        a = np.zeros(order + 1)
        a[0] = 1.0
        e = autocorr[0]
        for j in range(1, order + 1):
            if abs(e) < 1e-10:
                a[j] = 0
            else:
                a[j] = -np.sum(autocorr[1:j] * a[j-1:0:-1]) / e
            e = (1 - a[j]**2) * e
        roots = np.roots(a)
        roots = [r for r in roots if np.imag(r) > 0]
        freqs = sorted([np.arctan2(np.imag(r), np.real(r)) * self.sample_rate / (2 * np.pi) for r in roots])
        while len(freqs) < 3:
            freqs.append(freqs[-1] if freqs else 2500.0)
        return freqs[:3]
    
    def extract_speech_rate(self, audio):
        # 过零率估算语速
        zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2.0 * len(audio))
        words_per_min = zcr * 60 * self.sample_rate / 1000
        return words_per_min / 60.0
    
    def extract_energy_entropy(self, audio):
        # 能量熵
        frames = self._frame_signal(audio, 400, 160)
        if len(frames) == 0:
            return 0.0
        energies = np.sum(frames**2, axis=1) + 1e-10
        probs = energies / np.sum(energies)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return entropy / (np.log2(len(probs) + 1) + 1e-10)
    
    def extract_all(self, audio):
        return {
            "f0": self.extract_fundamental_frequency(audio),
            "formants": self.extract_formants(audio),
            "speech_rate": self.extract_speech_rate(audio),
            "energy_entropy": self.extract_energy_entropy(audio)
        }
    
    def _frame_signal(self, signal_array, frame_length, frame_shift):
        n_frames = 1 + (len(signal_array) - frame_length) // frame_shift
        if n_frames <= 0:
            return np.zeros((1, frame_length))
        frames = np.zeros((n_frames, frame_length))
        for i in range(n_frames):
            start = i * frame_shift
            end = start + frame_length
            if end <= len(signal_array):
                frames[i] = signal_array[start:end]
            else:
                frames[i, :len(signal_array)-start] = signal_array[start:]
        return frames
