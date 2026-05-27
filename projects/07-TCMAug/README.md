# TCMAug: GAN-based Tongue Image Augmentation

## 项目简介

TCMAug 是一个基于深度卷积生成对抗网络（DCGAN）的舌象数据增强系统，用于解决中医舌诊中数据稀缺的问题。

## 核心功能

- **DCGAN生成器**: 从100维随机噪声生成128x128x3的舌象图像
- **条件生成**: 支持按舌色（淡白舌、红舌、绛舌、紫舌）条件生成
- **判别器**: 评估生成图像的真实度
- **质量评估**: 多维度评估生成图像的质量（颜色一致性、舌形、舌苔可见度、无伪影）
- **风格插值**: 支持在不同舌象风格间插值

## 文件结构

07-TCMAug/
- dcgan_generator.py    # DCGAN生成器和风格编码器
- discriminator.py      # 判别器和质量评估器
- augmentation_pipeline.py  # 完整数据增强流水线
- demo.py               # 演示脚本
- requirements.txt      # 依赖
- README.md            # 本文件

## 快速开始

pip install numpy
python demo.py

## 技术特点

1. **简化DCGAN架构**: 使用numpy实现，适合快速验证
2. **多维度质量控制**: 生成后自动筛选高质量样本
3. **多样性保证**: 循环遍历不同舌象类型确保分布均匀

## 扩展方向

- 替换为PyTorch/TensorFlow实现真实DCGAN
- 增加更多舌象类型（裂纹舌、齿痕舌等）
- 添加Wasserstein GAN提升训练稳定性
- 集成到实际中医诊断系统
