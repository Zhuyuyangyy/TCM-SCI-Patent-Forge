"""
TCM-Acupuncture-Robot E01
机械臂推拿力度-穴位-形变闭环控制
 acupoint_locator.py - 20+穴位坐标库
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class Acupoint:
    """穴位数据结构"""
    name: str              # 穴位名称(中)
    name_en: str           # 英文名
    channel: str           # 所属经络
    location: str          # 解剖位置描述
    coordinates: Tuple[float, float, float]  # (x, y, z) 笛卡尔坐标 mm
    depth: Tuple[float, float]  # 安全深度范围 (min, max) mm
    risk_level: int        # 风险等级 1-5 (1=最低)
    contraindication: str  # 禁忌症


class AcupointDatabase:
    """中医穴位坐标数据库 - 支持20+穴位"""

    def __init__(self):
        self._acupoints: Dict[str, Acupoint] = {}
        self._initialize_database()

    def _initialize_database(self):
        """初始化20+穴位坐标库"""
        
        # ========== 足阳明胃经 ==========
        self.add_acupoint(Acupoint(
            name="足三里",
            name_en="Zusanli (ST36)",
            channel="足阳明胃经",
            location="犊鼻下3寸，胫骨前缘旁开1横指",
            coordinates=(120.0, -45.0, -380.0),  # 膝眼下方，胫骨前缘
            depth=(15.0, 35.0),
            risk_level=2,
            contraindication="空腹或过饱时慎用"
        ))

        self.add_acupoint(Acupoint(
            name="上巨虚",
            name_en="Shangjuxu (ST37)",
            channel="足阳明胃经",
            location="足三里下3寸",
            coordinates=(120.0, -45.0, -410.0),
            depth=(15.0, 30.0),
            risk_level=2,
            contraindication="避开血管"
        ))

        self.add_acupoint(Acupoint(
            name="下巨虚",
            name_en="Xiajuxu (ST39)",
            channel="足阳明胃经",
            location="上巨虚下3寸",
            coordinates=(120.0, -45.0, -440.0),
            depth=(15.0, 30.0),
            risk_level=2,
            contraindication="避开神经"
        ))

        self.add_acupoint(Acupoint(
            name="丰隆",
            name_en="Fenglong (ST40)",
            channel="足阳明胃经",
            location="外踝尖上8寸，胫骨前缘旁开2横指",
            coordinates=(130.0, -40.0, -460.0),
            depth=(15.0, 25.0),
            risk_level=2,
            contraindication="避开腓神经"
        ))

        # ========== 足太阳膀胱经 ==========
        self.add_acupoint(Acupoint(
            name="肾俞",
            name_en="Shenshu (BL23)",
            channel="足太阳膀胱经",
            location="第2腰椎棘突旁开1.5寸",
            coordinates=(0.0, 80.0, -150.0),
            depth=(20.0, 40.0),
            risk_level=3,
            contraindication="肾区叩击痛者禁用"
        ))

        self.add_acupoint(Acupoint(
            name="胃俞",
            name_en="Weishu (BL21)",
            channel="足太阳膀胱经",
            location="第12胸椎棘突旁开1.5寸",
            coordinates=(0.0, 80.0, -200.0),
            depth=(15.0, 35.0),
            risk_level=3,
            contraindication="背部皮肤病者慎用"
        ))

        self.add_acupoint(Acupoint(
            name="委中",
            name_en="Weizhong (BL40)",
            channel="足太阳膀胱经",
            location="腘横纹中点，股二头肌腱内侧",
            coordinates=(80.0, -20.0, -350.0),
            depth=(10.0, 25.0),
            risk_level=2,
            contraindication="腘窝血管神经丰富"
        ))

        self.add_acupoint(Acupoint(
            name="承山",
            name_en="Chengshan (BL57)",
            channel="足太阳膀胱经",
            location="腓肠肌腹下，伸足时肌腹下方凹陷",
            coordinates=(100.0, -35.0, -420.0),
            depth=(10.0, 20.0),
            risk_level=2,
            contraindication="跟腱断裂史者禁用"
        ))

        # ========== 足太阴脾经 ==========
        self.add_acupoint(Acupoint(
            name="三阴交",
            name_en="Sanyinjiao (SP6)",
            channel="足太阴脾经",
            location="内踝尖上3寸，胫骨内侧后缘",
            coordinates=(-120.0, -40.0, -395.0),
            depth=(15.0, 30.0),
            risk_level=3,
            contraindication="孕妇禁用，月经期慎用"
        ))

        self.add_acupoint(Acupoint(
            name="阴陵泉",
            name_en="Yinlingquan (SP9)",
            channel="足太阴脾经",
            location="胫骨内侧髁下缘凹陷处",
            coordinates=(-115.0, -35.0, -370.0),
            depth=(15.0, 25.0),
            risk_level=2,
            contraindication="血友病者慎用"
        ))

        self.add_acupoint(Acupoint(
            name="血海",
            name_en="Xuehai (SP10)",
            channel="足太阴脾经",
            location="髌骨内上缘上2寸，股内侧肌内侧",
            coordinates=(-90.0, 20.0, -320.0),
            depth=(10.0, 20.0),
            risk_level=2,
            contraindication="出血倾向者禁用"
        ))

        # ========== 足少阴肾经 ==========
        self.add_acupoint(Acupoint(
            name="涌泉",
            name_en="Yongquan (KI1)",
            channel="足少阴肾经",
            location="足底前1/3凹陷处",
            coordinates=(0.0, -50.0, -500.0),
            depth=(5.0, 15.0),
            risk_level=4,
            contraindication="足底溃疡感染者禁用"
        ))

        self.add_acupoint(Acupoint(
            name="太溪",
            name_en="Taixi (KI3)",
            channel="足少阴肾经",
            location="内踝后方，跟腱内侧凹陷",
            coordinates=(-110.0, -45.0, -390.0),
            depth=(10.0, 20.0),
            risk_level=3,
            contraindication="踝部骨折者慎用"
        ))

        self.add_acupoint(Acupoint(
            name="照海",
            name_en="Zhaohai (KI6)",
            channel="足少阴肾经",
            location="内踝下缘下方凹陷处",
            coordinates=(-105.0, -48.0, -388.0),
            depth=(5.0, 15.0),
            risk_level=3,
            contraindication="踝部急性损伤慎用"
        ))

        # ========== 足少阳胆经 ==========
        self.add_acupoint(Acupoint(
            name="阳陵泉",
            name_en="Yanglingquan (GB34)",
            channel="足少阳胆经",
            location="腓骨小头前下方凹陷处",
            coordinates=(130.0, -30.0, -360.0),
            depth=(15.0, 30.0),
            risk_level=2,
            contraindication="腓总神经损伤者慎用"
        ))

        self.add_acupoint(Acupoint(
            name="环跳",
            name_en="Huantiao (GB30)",
            channel="足少阳胆经",
            location="股骨大转子与骶管裂孔连线外1/3处",
            coordinates=(150.0, 30.0, -280.0),
            depth=(40.0, 80.0),
            risk_level=4,
            contraindication="髋关节结核、肿瘤者禁用"
        ))

        self.add_acupoint(Acupoint(
            name="风市",
            name_en="Fengshi (GB31)",
            channel="足少阳胆经",
            location="大腿外侧，髌骨上7寸，股外侧肌上",
            coordinates=(140.0, 25.0, -330.0),
            depth=(20.0, 40.0),
            risk_level=3,
            contraindication="股外侧皮神经损伤者慎用"
        ))

        # ========== 任脉 ==========
        self.add_acupoint(Acupoint(
            name="关元",
            name_en="Guanyuan (CV4)",
            channel="任脉",
            location="脐下3寸，腹正中线",
            coordinates=(0.0, -60.0, 50.0),
            depth=(20.0, 50.0),
            risk_level=3,
            contraindication="孕妇禁用，膀胱充盈时慎用"
        ))

        self.add_acupoint(Acupoint(
            name="中脘",
            name_en="Zhongwan (CV12)",
            channel="任脉",
            location="脐上4寸，腹正中线",
            coordinates=(0.0, -60.0, 100.0),
            depth=(15.0, 40.0),
            risk_level=3,
            contraindication="胃溃疡、肿瘤者慎用"
        ))

        # ========== 足阳明胃经(续) ==========
        self.add_acupoint(Acupoint(
            name="梁丘",
            name_en="Liangqiu (ST34)",
            channel="足阳明胃经",
            location="髌骨外上缘上2寸，股直肌外侧",
            coordinates=(100.0, 15.0, -310.0),
            depth=(10.0, 20.0),
            risk_level=2,
            contraindication="股神经损伤者慎用"
        ))

        self.add_acupoint(Acupoint(
            name="犊鼻",
            name_en="Dubi (ST35)",
            channel="足阳明胃经",
            location="髌骨下缘，髌韧带外侧凹陷",
            coordinates=(110.0, -10.0, -375.0),
            depth=(8.0, 15.0),
            risk_level=2,
            contraindication="膝关节肿胀积液者慎用"
        ))

        # ========== 经外奇穴 ==========
        self.add_acupoint(Acupoint(
            name="胆囊穴",
            name_en="Dannangxue (EX-LE6)",
            channel="经外奇穴",
            location="阳陵泉下1-2寸，压痛点",
            coordinates=(125.0, -32.0, -370.0),
            depth=(15.0, 25.0),
            risk_level=3,
            contraindication="胆道感染者慎用"
        ))

        self.add_acupoint(Acupoint(
            name="阑尾穴",
            name_en="Lanweixue (EX-LE7)",
            channel="经外奇穴",
            location="足三里下1-2寸，压痛点",
            coordinates=(118.0, -46.0, -390.0),
            depth=(15.0, 25.0),
            risk_level=3,
            contraindication="阑尾炎急性期禁按"
        ))

    def add_acupoint(self, acupoint: Acupoint):
        """添加穴位到数据库"""
        self._acupoints[acupoint.name] = acupoint
        self._acupoints[acupoint.name_en.split()[0]] = acupoint  # 英文简称

    def get_acupoint(self, name: str) -> Optional[Acupoint]:
        """根据名称查找穴位"""
        return self._acupoints.get(name)

    def list_all_acupoints(self) -> List[Acupoint]:
        """列出所有穴位"""
        seen = set()
        result = []
        for ap in self._acupoints.values():
            if ap.name not in seen:
                seen.add(ap.name)
                result.append(ap)
        return sorted(result, key=lambda x: x.name)

    def get_by_channel(self, channel: str) -> List[Acupoint]:
        """按经络查找穴位"""
        return [ap for ap in self._acupoints.values() 
                if ap.channel == channel and ap.name not in [x.name for x in []]]

    def get_by_risk_level(self, level: int) -> List[Acupoint]:
        """按风险等级筛选"""
        return [ap for ap in self._acupoints.values() 
                if ap.risk_level <= level]

    def calculate_distance(self, p1: Tuple[float, float, float], 
                          p2: Tuple[float, float, float]) -> float:
        """计算两点间距离 (mm)"""
        return math.sqrt(sum((a-b)**2 for a, b in zip(p1, p2)))

    def find_nearest(self, position: Tuple[float, float, float], 
                     max_distance: float = 50.0) -> Optional[Tuple[Acupoint, float]]:
        """查找最近的穴位"""
        nearest = None
        min_dist = max_distance
        
        for ap in self._acupoints.values():
            if ap.name not in [x.name for x in []]:
                dist = self.calculate_distance(position, ap.coordinates)
                if dist < min_dist:
                    min_dist = dist
                    nearest = (ap, dist)
        
        return nearest


def main():
    """测试穴位数据库"""
    db = AcupointDatabase()
    
    print("=" * 60)
    print("TCM 穴位坐标数据库 - 共{}个穴位".format(len(db.list_all_acupoints())))
    print("=" * 60)
    
    # 列出所有穴位
    print("\n【所有穴位列表】")
    for i, ap in enumerate(db.list_all_acupoints(), 1):
        print(f"{i:2d}. {ap.name:<6s} ({ap.name_en:<20s}) | {ap.channel:<12s} | 风险:{ap.risk_level}")
    
    # 测试查找足三里
    print("\n【查找足三里】")
    zusanli = db.get_acupoint("足三里")
    if zusanli:
        print(f"  名称: {zusanli.name}")
        print(f"  英文: {zusanli.name_en}")
        print(f"  经络: {zusanli.channel}")
        print(f"  位置: {zusanli.location}")
        print(f"  坐标: {zusanli.coordinates}")
        print(f"  安全深度: {zusanli.depth}")
        print(f"  风险等级: {zusanli.risk_level}")
        print(f"  禁忌: {zusanli.contraindication}")
    
    # 测试距离计算
    print("\n【距离计算测试】")
    pos = (120.0, -44.0, -379.0)
    result = db.find_nearest(pos)
    if result:
        ap, dist = result
        print(f"  当前位置: {pos}")
        print(f"  最近穴位: {ap.name} ({dist:.2f}mm)")
    
    # 膀胱经穴位
    print("\n【足太阳膀胱经穴位】")
    for ap in db.get_by_channel("足太阳膀胱经"):
        print(f"  - {ap.name}: {ap.coordinates}")


if __name__ == "__main__":
    main()
