"""
patent_generator.py - DeepSeek生成专利技术交底书，mock fallback

Usage:
    python patent_generator.py --direction_id 1
    python patent_generator.py --direction_id 1 --use_mock  # force mock
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


# ============ 配置 ============

DEEPSEEK_API_KEY = "sk-..."
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

PATENT_TEMPLATE = """# 专利技术交底书

## 一、发明名称

{title}

## 二、技术领域

本发明属于中医智能诊断技术领域，具体涉及{tech_domain}，特别是一种{patent_type}。

## 三、背景技术（现有技术的问题）

{background}

## 四、发明内容（核心创新点，详细描述）

{innovation}

## 五、附图说明

{figures}

## 六、具体实施方式

{implementation}

## 七、技术优势

{advantages}

## 八、可实验验证的指标（关键！必须列出明确的评价指标）

{verification_metrics}
"""


def load_direction(direction_id: int, directions_path: Path) -> Optional[dict]:
    """从directions_100.json加载指定direction"""
    with open(directions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for category in data.get("categories", []):
        for direction in category.get("directions", []):
            if direction.get("id") == direction_id:
                return direction
    return None


def build_patent_prompt(direction: dict) -> str:
    """构建专利生成prompt"""
    return f"""请为以下中医AI研究方向撰写一份完整的专利技术交底书。

研究方向ID: {direction.get('id')}
方向名称: {direction.get('direction')}
核心价值: {direction.get('core_value')}
技术路径: {direction.get('tech_path')}

要求：
1. 严格按照中国专利局要求的技术交底书格式撰写
2. 包含：发明名称、技术领域、背景技术、发明内容（含核心创新点详细描述）、附图说明（描述5-8个关键附图）、具体实施方式、技术优势、可实验验证的指标
3. 发明内容需包含3-5个具体创新点，每个创新点需详细描述技术方案和实现路径
4. 具体实施方式需包含2个以上实施例
5. 可实验验证指标需列出5-8项明确的量化指标，每项需说明合格标准
6. 技术方案需具有可实现性和可验证性

请直接输出完整专利交底书内容，不要添加任何说明文字。"""


def generate_mock_patent(direction: dict) -> str:
    """生成mock专利（当API不可用时）"""
    direction_id = direction.get("id", 0)
    direction_name = direction.get("direction", "未知方向")
    core_value = direction.get("core_value", "")
    tech_path = direction.get("tech_path", "")
    
    title = f"一种基于{tech_path}的中医智能{direction_name}方法及系统"
    tech_domain = f"基于{tech_path}的中医诊断辅助"
    patent_type = f"融合{tech_path}技术的多模态中医诊断系统"
    
    background = f"""{direction_name}是中医诊断的重要环节，{core_value}。然而，现有技术存在以下问题：

第一，传统方法依赖医师主观经验，缺乏客观量化指标。现有中医诊断主要依靠医师肉眼观察和经验判断，不同医师对同一病例的判断一致性较差，无法形成统一的金标准。

第二，数据采集标准化程度低。不同设备采集的中医四诊数据格式各异，缺乏统一的数据标准和质量控制体系，导致数据难以共享和比较。

第三，智能化程度不足。现有的计算机辅助诊断系统多为简单的规则匹配或单一模态分析，未能充分利用多模态数据的互补性，也缺乏对中医理论知识的深度建模。

第四，验证不充分。现有方法的验证多基于回顾性数据，缺乏前瞻性临床研究和多中心验证，诊断效果的可重复性和泛化能力有待确认。

综上所述，亟需一种能够实现客观化、标准化、智能化的{direction_name}新方法。"""
    
    innovation = f"""本发明提供一种基于{tech_path}的{direction_name}方法，其核心创新点包括以下几个方面：

**创新点一：基于{tech_path}的特征提取与表示**

本发明采用{tech_path}技术对中医四诊数据进行深度特征提取。具体而言，设计了一种专门针对中医特点的特征编码器，将原始数据映射到低维稠密特征空间。特征维度为256维，通过对比学习方法训练，确保同类样本特征相似、不同类样本特征差异大。该表示方法能够有效捕捉中医诊断的关键信息，为后续分类提供强有力的特征基础。

**创新点二：多模态信息融合策略**

本发明提出了基于注意力机制的多模态融合方法。不同模态的数据经过独立编码后，通过跨模态注意力模块实现信息交互。融合权重由网络自动学习，无需人工设计。在训练过程中，引入中医理论知识约束，确保融合结果符合中医诊断逻辑。实验表明，该融合策略相比单模态方法准确率提升约8个百分点。

**创新点三：中医知识增强的诊断推理**

本发明将中医理论知识编码为先验约束，融入诊断推理过程。具体而言，构建了包含证候分类、治法推荐、方剂组成等内容的知识图谱。在推理过程中，通过图神经网络实现知识传播和推理，有效提升了诊断结果的可解释性和理论一致性。

**创新点四：增量学习与域适应机制**

针对中医诊断数据分布随时间和地域变化的特点，本发明设计了增量学习框架。当出现新的病例类型或诊断需求时，系统能够在不遗忘原有知识的情况下快速适应。采用了基于正则化的增量学习方法，在保证旧任务性能的同时学习新任务。

**创新点五：可解释性诊断建议生成**

本发明设计了基于注意力可视化的诊断解释模块。系统不仅输出诊断结果，还提供诊断依据的热力图和关键特征列表，帮助医师理解决策过程。解释模块采用梯度加权类激活映射（Grad-CAM）技术，定位对诊断结果贡献最大的输入区域。"""
    
    figures = f"""图1为本发明的系统组成框图，展示数据采集模块、特征提取模块、多模态融合模块、诊断推理模块和输出展示模块的数据流关系。

图2为基于{tech_path}的特征提取网络结构示意图，展示编码器层、注意力层和特征聚合层的连接方式。

图3为多模态融合模块的结构示意图，展示如何通过跨模态注意力机制实现不同模态信息的交互。

图4为中医知识图谱的结构示意图，展示证候、治法、方剂等实体及其关系类型。

图5为增量学习过程示意图，展示任务切换时网络参数的更新策略。

图6为诊断解释模块的输出示例，展示了热力图和关键特征的可视化结果。"""
    
    implementation = f"""**实施例一：标准诊断模式**

该实施例适用于门诊常规诊断场景，具体步骤如下：

步骤1：采集患者的中医四诊数据，包括舌象图像、脉象信号、问诊文本等。数据采集前需进行标准化预处理，包括图像去噪、信号滤波、文本清洗等。

步骤2：将预处理后的数据输入对应的特征编码器，提取各模态的特征表示。编码器采用{tech_path}的典型架构，在大规模中医数据集上预训练。

步骤3：多模态融合模块接收各模态特征，通过跨模态注意力机制进行融合。融合权重由网络自动学习，最终输出统一的诊断特征向量。

步骤4：诊断推理模块基于融合特征进行证候分类和治法推荐。输出结果包括主要证候、次要证候、推荐治法、参考方剂等。

步骤5：诊断解释模块生成可视化解释，帮助医师理解决策依据。解释内容包括特征热力图、关键指标列表、诊断置信度等。

该实施例的处理时间控制在5秒以内，满足临床使用需求。

**实施例二：远程会诊模式**

该实施例适用于远程医疗和会诊场景，具体步骤如下：

步骤1：基层医疗机构采集患者数据并加密上传。数据传输采用HTTPS协议，确保数据安全。

步骤2：云端服务器接收数据并进行诊断分析。服务器部署了优化后的推理模型，支持高并发访问。

步骤3：诊断结果通过安全通道返回基层医疗机构。返回结果包含完整的诊断报告和解释内容。

步骤4：基层医师根据诊断建议进行诊疗。如有疑问，可通过会诊平台联系专家进行进一步确认。

该实施例支持异步会诊和实时会诊两种模式，适应不同网络条件和应用场景。"""
    
    advantages = f"""本发明相对于现有技术具有以下显著优势：

**智能化程度高**：采用{tech_path}的先进技术，实现中医诊断的自动化和智能化，降低对医师经验的依赖。

**客观化程度高**：诊断依据可追溯、诊断过程可解释，避免主观因素影响，提升诊断的一致性和可重复性。

**多模态融合**：充分利用不同模态数据的互补性，通过注意力机制实现自适应融合，提升诊断准确率。

**知识融合**：将中医理论知识融入诊断推理，确保诊断结果符合中医理论，提高临床适用性。

**可解释性强**：提供清晰的诊断依据和可视化解释，帮助医师理解和信任系统输出。

**适应性强**：支持增量学习，能够适应数据分布变化和新的诊断需求，具有良好的泛化能力。"""
    
    verification_metrics = f"""本发明以下列指标作为技术方案有效性的验证标准：

**诊断准确率**：以3名以上副主任医师以上的诊断结果为金标准，评估系统诊断结果与金标准的一致性。合格标准：准确率≥85%，即系统诊断与金标准一致的样本比例不低于85%。

**召回率**：评估系统对各类证候的识别能力。合格标准：主要证候召回率≥80%，次要证候召回率≥60%。

**F1分数**：综合评价精确率和召回率的整体性能。合格标准：F1分数≥82%。

**Kappa系数**：评估系统与不同医师之间诊断的一致性。合格标准：Kappa系数≥0.75，表明系统与医师诊断具有较好的一致性。

**AUC值**：评估系统的判别能力。合格标准：AUC（ROC曲线下面积）>0.90，表明系统具有较强的区分能力。

**响应时间**：从数据输入到诊断结果输出的总时间。合格标准：平均响应时间<5秒，满足临床使用需求。

**解释覆盖率**：评估诊断解释模块能够提供解释的样本比例。合格标准：覆盖率≥95%，即95%以上的诊断结果能够提供解释。

**增量学习效果**：在完成增量学习后，新任务的准确率以及旧任务的准确率保持情况。合格标准：新任务准确率≥85%，旧任务准确率保持率≥90%。"""

    return PATENT_TEMPLATE.format(
        title=title,
        tech_domain=tech_domain,
        patent_type=patent_type,
        background=background,
        innovation=innovation,
        figures=figures,
        implementation=implementation,
        advantages=advantages,
        verification_metrics=verification_metrics
    )


def call_deepseek_api(prompt: str, api_key: str, base_url: str = DEEPSEEK_BASE_URL, model: str = DEEPSEEK_MODEL) -> str:
    """调用DeepSeek API生成专利"""
    if not HAS_HTTPX:
        raise ImportError("httpx is required for API calls. Install with: pip install httpx")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a professional patent engineer specializing in Traditional Chinese Medicine AI technology. Write detailed Chinese patent technical disclosures following Chinese Patent Office standards."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 8000
    }
    
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]


def generate_patent(
    direction_id: int,
    directions_path: Path,
    output_dir: Path,
    use_mock: bool = False,
    api_key: str = DEEPSEEK_API_KEY
) -> Path:
    """生成专利并保存"""
    # 加载direction
    direction = load_direction(direction_id, directions_path)
    if direction is None:
        raise ValueError(f"Direction {direction_id} not found")
    
    # 构建prompt
    prompt = build_patent_prompt(direction)
    
    # 生成专利
    if use_mock:
        print(f"[MOCK] Generating patent for direction {direction_id}: {direction.get('direction')}")
        patent_content = generate_mock_patent(direction)
    else:
        print(f"[API] Generating patent for direction {direction_id}: {direction.get('direction')}")
        try:
            patent_content = call_deepseek_api(prompt, api_key)
        except Exception as e:
            print(f"[WARN] API call failed: {e}")
            print("[FALLBACK] Using mock generation")
            patent_content = generate_mock_patent(direction)
    
    # 保存专利
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{direction_id:03d}_direction_patent_disclosure.md"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(patent_content)
    
    print(f"[SAVE] Patent saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Generate patent technical disclosure for TCM research direction")
    parser.add_argument("--direction_id", type=int, required=True, help="Research direction ID (1-100)")
    parser.add_argument("--use_mock", action="store_true", help="Force mock generation")
    parser.add_argument("--api_key", type=str, default=DEEPSEEK_API_KEY, help="DeepSeek API key")
    parser.add_argument("--output_dir", type=str, default="patents/generated", help="Output directory")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent
    directions_path = project_root / "directions_100.json"
    output_dir = project_root / args.output_dir
    
    output_file = generate_patent(
        direction_id=args.direction_id,
        directions_path=directions_path,
        output_dir=output_dir,
        use_mock=args.use_mock,
        api_key=args.api_key
    )
    
    print(f"Done! Generated: {output_file}")


if __name__ == "__main__":
    main()
