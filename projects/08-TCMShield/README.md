# TCMShield: TCM Multi-Agent Safety Defense System

## 项目简介

TCMShield 是一个中医多智能体安全防御系统，为TCM Agent提供安全保障，检测和拦截可能导致危险的输出。

## 核心功能

- **安全策略引擎**: 定义中药剂量上限、配伍禁忌（十八反、十九畏）、年龄适宜性等规则
- **威胁检测器**: 检测Agent输出中的超剂量、禁忌配伍、年龄不适、适应症矛盾等问题
- **干预引擎**: 根据威胁严重度决定allow/warn/block动作
- **Shield包装器**: 透明包装任意TCM Agent，提供无缝安全防护

## 文件结构

08-TCMShield/
- agent_shield.py      # 核心防御系统
- mock_tcm_agent.py    # 测试用模拟Agent
- demo.py              # 演示脚本
- requirements.txt     # 依赖
- README.md           # 本文件

## 快速开始

pip install numpy
python demo.py

## 安全检测类型

1. **超剂量检测**: 检查单味药剂量是否超过安全上限
2. **配伍禁忌检测**: 检查是否违反十八反、十九畏
3. **年龄适宜性**: 检查药物是否适合患者年龄
4. **适应症矛盾**: 检查诊断与患者禁忌症是否冲突

## 扩展方向

- 集成更多中药安全数据
- 添加药物相互作用检测
- 支持自定义安全策略
- 与真实TCM诊断系统集成
