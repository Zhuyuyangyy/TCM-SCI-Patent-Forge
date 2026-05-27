
"""AudioSense-TCM 完整演示（纯NumPy）"""
import numpy as np
from audio_extractor import AudioFeatureExtractor
from zangfu_mapper import ZangFuMapper

def synthetic_audio_features():
    return np.random.randn(128).astype(np.float32)

def run_demo():
    print("=" * 60)
    print("AudioSense-TCM Demo: 语音韵律-脏腑功能映射")
    print("=" * 60)
    print()
    print("[Step 1] 语音特征提取...")
    extractor = AudioFeatureExtractor()
    audio_features = synthetic_audio_features()
    print(f"  特征维度: {audio_features.shape}")
    f0_range = (80 + abs(audio_features[0])*200, 200 + abs(audio_features[1])*200)
    print(f"  基频范围: {f0_range[0]:.1f}-{f0_range[1]:.1f} Hz")
    print()
    print("[Step 2] 脏腑功能评估...")
    np.random.seed(42)
    latent = np.random.randn(4).astype(np.float32)
    organ_probs = np.random.rand(5).astype(np.float32)
    organ_probs = organ_probs / organ_probs.sum()
    print(f"  潜空间维度: {latent.shape}")
    print()
    print("[Step 3] 脏腑功能评估报告...")
    mapper = ZangFuMapper()
    report = mapper.map_to_zangfu(latent, organ_probs)
    for item in report:
        print(f"  {item["organ"]}（{item["status"]}）: {item["probability"]:.3f}")
        print(f"    {item["tcm_interpretation"]}")
    print()
    print("=" * 60)
    print("Demo 完成!")

if __name__ == "__main__":
    run_demo()
