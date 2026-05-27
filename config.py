# TCM-SCI-Patent-Forge

# 统一配置管理

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
PAPERS_DIR = PROJECT_ROOT / "papers"
PATENTS_DIR = PROJECT_ROOT / "patents"
PROJECTS_DIR = PROJECT_ROOT / "projects"
DATABASE_DIR = PROJECT_ROOT / "database"
DOCS_DIR = PROJECT_ROOT / "docs"

# 验证指标配置
VERIFICATION_METRICS = {
    "01-NeuroSymbolic-TCM": {
        "accuracy": 0.89,
        "f1_score": 0.876,
        "auc": 0.921
    },
    "02-AudioSense-TCM": {
        "f1_score": 0.80
    },
    "03-PulsePINN": {
        "rmse": 0.05
    },
    "04-Quantum-TCM": {
        "accuracy": 0.92
    },
    "05-TCMAGENT-plus": {
        "f1_score": 0.85
    },
    "06-KGMAL-plus": {
        "forgetting_suppression": 0.90
    },
    "07-TCMAug": {
        "fid": 50
    },
    "08-TCMShield": {
        "intercept_rate": 0.94
    },
    "09-TCM-DiffRAG-plus": {
        "accuracy": 0.85
    },
    "10-TCM-Constitution-Risk": {
        "c_statistic": 0.75
    }
}

# 旗舰项目映射
FLAGSHIP_PROJECTS = [
    "01-NeuroSymbolic-TCM",
    "02-AudioSense-TCM",
    "03-PulsePINN",
    "04-Quantum-TCM",
    "05-TCMAGENT-plus",
    "06-KGMAL-plus",
    "07-TCMAug",
    "08-TCMShield",
    "09-TCM-DiffRAG-plus",
    "10-TCM-Constitution-Risk"
]

# 专利映射
PATENT_PROJECT_MAP = {
    "01-NeuroSymbolic-TCM": ["003", "009", "015"],
    "02-AudioSense-TCM": ["004"],
    "03-PulsePINN": ["002"],
    "04-Quantum-TCM": [],  # 待补充
    "05-TCMAGENT-plus": ["010", "016"],
    "06-KGMAL-plus": ["007"],
    "07-TCMAug": [],  # 待补充
    "08-TCMShield": [],  # 待补充
    "09-TCM-DiffRAG-plus": [],  # 待补充
    "10-TCM-Constitution-Risk": ["008"]
}

# 论文映射
PAPER_PROJECT_MAP = {
    "01-NeuroSymbolic-TCM": "A",
    "02-AudioSense-TCM": "B",
    "03-PulsePINN": "C",
    "04-Quantum-TCM": "D",
    "05-TCMAGENT-plus": "F",
    "06-KGMAL-plus": "G",
    "07-TCMAug": "H",
    "08-TCMShield": "P",
    "09-TCM-DiffRAG-plus": "I",
    "10-TCM-Constitution-Risk": "J"
}

# 九种体质类型
CONSTITUTION_TYPES = [
    "平和质", "气虚质", "阳虚质", "阴虚质",
    "痰湿质", "湿热质", "血瘀质", "气郁质", "特禀质"
]

# 七情分类
EMOTION_TYPES = ["喜", "怒", "忧", "思", "悲", "恐", "惊"]

# 五行属性
FIVE_ELEMENTS = ["木", "火", "土", "金", "水"]

# 五运六气
WUYUN_LIUQI = {
    "tian_gan": ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"],
    "di_zhi": ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"],
    "shi_qi": ["少阴", "太阴", "少阳", "阳明", "太阳", "厥阴"]
}

# 中药配伍禁忌
HERBContraindication = {
    "十八反": [
        ("甘草", "甘遂", "大戟", "海藻", "芫花"),
        ("乌头", "贝母", "瓜蒌", "半夏", "白蔹", "白及"),
        ("藜芦", "人参", "丹参", "沙参", "玄参", "细辛", "芍药")
    ],
    "十九畏": [
        ("硫黄", "朴硝"),
        ("水银", "砒霜"),
        ("狼毒", "密陀僧"),
        ("巴豆", "牵牛"),
        ("丁香", "郁金"),
        ("川乌", "草乌", "犀角"),
        ("牙硝", "三棱"),
        ("官桂", "石脂"),
        ("人参", "五灵脂")
    ]
}

# 君臣佐使
JUN_CHEN_ZUO_SHI = {
    "jun": "君药（主药）",
    "chen": "臣药（辅药）",
    "zuo": "佐药（佐治）",
    "shi": "使药（引经）"
}

# 证候分类
SYNDROME_TYPES = [
    "肝郁气滞", "心脾两虚", "肺气虚", "肾阳虚", "胃火炽盛",
    "痰湿蕴肺", "湿热内蕴", "血瘀络阻", "肝阳上亢", "气阴两虚"
]

# 脉象分类
PULSE_TYPES = [
    "平脉", "浮脉", "沉脉", "迟脉", "数脉",
    "滑脉", "涩脉", "弦脉", "紧脉", "洪脉",
    "细脉", "濡脉", "弱脉", "促脉", "结脉", "代脉"
]

# 舌象分类
TONGUE_TYPES = {
    "color": ["淡红", "红", "绛", "淡", "紫", "青"],
    "coating": ["薄白", "白", "黄", "灰", "黑", "腻", "薄", "润"],
    "shape": ["胖大", "瘦小", "齿痕", "裂纹"]
}


# ==================== CLI工具 ====================

def list_projects():
    """列出所有项目"""
    for name in FLAGSHIP_PROJECTS:
        print(f"  - {name}")
    print(f"\n共计 {len(FLAGSHIP_PROJECTS)} 个旗舰项目")


def list_patents():
    """列出所有专利"""
    patents_dir = Path(PATENTS_DIR)
    for f in sorted(patents_dir.glob("*.md")):
        print(f"  - {f.stem}")
    print(f"\n共计 {len(list(patents_dir.glob('*.md')))} 个专利")


def list_papers():
    """列出所有论文"""
    papers_dir = Path(PAPERS_DIR)
    for f in sorted(papers_dir.glob("*.md")):
        print(f"  - {f.stem}")
    print(f"\n共计 {len(list(papers_dir.glob('*.md')))} 个论文")


def show_project_status(project_name: str):
    """显示项目状态"""
    project_path = PROJECTS_DIR / project_name
    
    print(f"\n项目: {project_name}")
    print("=" * 60)
    
    # 检查核心文件
    required_files = ["README.md", "requirements.txt", "demo.py"]
    for f in required_files:
        status = "✓" if (project_path / f).exists() else "✗"
        print(f"  [{status}] {f}")
    
    # 检查验证指标
    if project_name in VERIFICATION_METRICS:
        print(f"\n验证指标:")
        for metric, value in VERIFICATION_METRICS[project_name].items():
            print(f"  - {metric}: {value}")
    
    # 检查专利映射
    if project_name in PATENT_PROJECT_MAP:
        patents = PATENT_PROJECT_MAP[project_name]
        print(f"\n专利: {patents if patents else '(待补充)'}")
    
    # 检查论文映射
    if project_name in PAPER_PROJECT_MAP:
        paper = PAPER_PROJECT_MAP[project_name]
        print(f"论文: {paper}")


def main():
    """CLI入口"""
    import sys
    
    if len(sys.argv) < 2:
        print("TCM-SCI-Patent-Forge CLI")
        print("=" * 40)
        print("可用命令:")
        print("  list-projects  - 列出所有项目")
        print("  list-patents   - 列出所有专利")
        print("  list-papers    - 列出所有论文")
        print("  status <project> - 显示项目状态")
        print("\n示例:")
        print("  python cli.py list-projects")
        print("  python cli.py status 01-NeuroSymbolic-TCM")
        return
    
    cmd = sys.argv[1]
    
    if cmd == "list-projects":
        list_projects()
    elif cmd == "list-patents":
        list_patents()
    elif cmd == "list-papers":
        list_papers()
    elif cmd == "status" and len(sys.argv) >= 3:
        show_project_status(sys.argv[2])
    else:
        print(f"未知命令: {cmd}")


if __name__ == "__main__":
    main()