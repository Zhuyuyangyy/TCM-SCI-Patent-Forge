"""
TCM-Acupuncture-Robot E01
机械臂推拿力度-穴位-形变闭环控制
 safety_monitor.py - 风险区域避让与安全监控
"""

from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import math


class RiskRegionType(Enum):
    """风险区域类型"""
    NEURAL_ZONE = "神经区"         # 神经密集区
    VASCULAR_ZONE = "血管区"       # 大血管区域
    JOINT_SPACE = "关节腔"         # 关节腔
    FRACTURE_LINE = "骨折线"       # 陈旧性骨折线
    CAROTID_ZONE = "颈动脉区"      # 颈动脉窦
    ABDOMINAL_ORGAN = "腹腔脏器"   # 腹腔脏器
    SPINAL_ZONE = "脊柱区"         # 脊髓/椎板
    PERIOSTEUM_NEAR = "骨膜附近"   # 过度接近骨膜


@dataclass
class RiskRegion:
    """风险区域定义"""
    region_type: RiskRegionType
    name: str
    center: Tuple[float, float, float]   # 中心坐标
    radius: float                        # 影响半径 mm
    severity: int                        # 严重程度 1-5
    description: str                     # 描述


@dataclass
class SafetyAlert:
    """安全警报"""
    timestamp: float
    alert_level: str                      # CRITICAL/DANGER/WARNING/INFO
    message: str
    distance: float                       # 到风险区域距离 mm
    recommended_action: str               # 建议措施


@dataclass
class SafetyStatus:
    """安全状态"""
    is_safe: bool
    distance_to_nearest_risk: float      # mm
    nearest_risk_name: str
    overall_risk_level: int              # 1-5
    alerts: List[SafetyAlert]
    operational_limit: str               # 操作限制说明


class SafetyMonitor:
    """安全监控器
    
    实现风险区域检测、避让和安全监控
    """
    
    # 默认风险区域数据库
    DEFAULT_RISK_REGIONS = [
        # 下肢风险区域
        RiskRegion(RiskRegionType.NEURAL_ZONE, "腓总神经",
                   (130.0, -30.0, -355.0), 15.0, 4,
                   "腓骨小头附近，压迫可致足下垂"),
        
        RiskRegion(RiskRegionType.VASCULAR_ZONE, "腘动脉",
                   (85.0, -15.0, -345.0), 12.0, 5,
                   "腘窝中央，破裂可致大出血"),
        
        RiskRegion(RiskRegionType.VASCULAR_ZONE, "股动脉",
                   (90.0, 10.0, -250.0), 15.0, 5,
                   "大腿前内侧，主要动脉"),
        
        RiskRegion(RiskRegionType.JOINT_SPACE, "膝关节腔",
                   (115.0, -5.0, -370.0), 10.0, 3,
                   "犊鼻穴下方，感染可致关节炎"),
        
        RiskRegion(RiskRegionType.NEURAL_ZONE, "隐神经",
                   (-100.0, -35.0, -320.0), 10.0, 3,
                   "内膝眼附近，皮神经损伤致麻木"),
        
        # 腹部风险区域
        RiskRegion(RiskRegionType.ABDOMINAL_ORGAN, "肾脏",
                   (-60.0, -70.0, -145.0), 25.0, 5,
                   "第2腰椎旁，肾区叩击痛禁用"),
        
        RiskRegion(RiskRegionType.ABDOMINAL_ORGAN, "肝脏",
                   (50.0, -70.0, 80.0), 30.0, 5,
                   "右肋弓下，重力按压可致损伤"),
        
        RiskRegion(RiskRegionType.ABDOMINAL_ORGAN, "脾脏",
                   (-50.0, -70.0, 80.0), 25.0, 5,
                   "左肋弓下，脾大时禁用"),
        
        # 脊柱区域
        RiskRegion(RiskRegionType.SPINAL_ZONE, "脊髓",
                   (0.0, 60.0, -200.0), 8.0, 5,
                   "椎管内，针刺过深可致截瘫"),
        
        RiskRegion(RiskRegionType.SPINAL_ZONE, "椎板",
                   (0.0, 75.0, -180.0), 10.0, 4,
                   "棘突旁，过深可进入椎管"),
        
        # 特殊穴位风险
        RiskRegion(RiskRegionType.PERIOSTEUM_NEAR, "骨膜边界-足三里",
                   (120.0, -45.0, -375.0), 5.0, 3,
                   "接近胫骨，深度>35mm有风险"),
    ]

    def __init__(self, risk_regions: Optional[List[RiskRegion]] = None):
        self._risk_regions = risk_regions or self.DEFAULT_RISK_REGIONS.copy()
        self._enabled = True
        self._alert_history: List[SafetyAlert] = []
        
    def enable(self):
        """启用监控"""
        self._enabled = True
        
    def disable(self):
        """禁用监控"""
        self._enabled = False

    def check_position(self, position: Tuple[float, float, float],
                      depth: float = 0.0,
                      timestamp: Optional[float] = None) -> SafetyStatus:
        """检查位置安全性
        
        Args:
            position: (x, y, z) 坐标 mm
            depth: 当前深度 mm
            timestamp: 时间戳
            
        Returns:
            安全状态
        """
        if timestamp is None:
            import time
            timestamp = time.time()
            
        if not self._enabled:
            return SafetyStatus(
                is_safe=True,
                distance_to_nearest_risk=float('inf'),
                nearest_risk_name="N/A",
                overall_risk_level=0,
                alerts=[],
                operational_limit="监控已禁用"
            )
        
        alerts: List[SafetyAlert] = []
        min_distance = float('inf')
        nearest_risk = None
        
        for region in self._risk_regions:
            dist = self._calculate_distance(position, region.center)
            
            # 考虑深度因素
            effective_radius = region.radius + depth * 0.1
            
            if dist < effective_radius:
                # 进入风险区域
                alert_level = self._get_alert_level(dist, effective_radius, region.severity)
                message = self._generate_alert_message(region, dist, depth)
                action = self._get_recommended_action(region, dist, depth)
                
                alert = SafetyAlert(
                    timestamp=timestamp,
                    alert_level=alert_level,
                    message=message,
                    distance=dist,
                    recommended_action=action
                )
                alerts.append(alert)
                self._alert_history.append(alert)
            
            if dist < min_distance:
                min_distance = dist
                nearest_risk = region
        
        # 确定整体风险等级
        if alerts and nearest_risk:
            risk_level = min(5, nearest_risk.severity)
        else:
            risk_level = 0
            
        is_safe = len([a for a in alerts if a.alert_level in ('CRITICAL', 'DANGER')]) == 0
        
        return SafetyStatus(
            is_safe=is_safe,
            distance_to_nearest_risk=min_distance,
            nearest_risk_name=nearest_risk.name if nearest_risk else "N/A",
            overall_risk_level=risk_level,
            alerts=alerts,
            operational_limit=self._get_operational_limit(alerts, min_distance)
        )

    def _calculate_distance(self, p1: Tuple[float, float, float],
                           p2: Tuple[float, float, float]) -> float:
        """计算两点距离"""
        return math.sqrt(sum((a-b)**2 for a, b in zip(p1, p2)))

    def _find_region(self, name: str) -> Optional[RiskRegion]:
        """根据名称查找风险区域"""
        for r in self._risk_regions:
            if r.name == name:
                return r
        return None

    def _get_alert_level(self, distance: float, radius: float, 
                        severity: int) -> str:
        """确定警报级别"""
        ratio = distance / radius if radius > 0 else 0
        
        if ratio < 0.3 or severity >= 5:
            return "CRITICAL"
        elif ratio < 0.6 or severity >= 4:
            return "DANGER"
        elif ratio < 1.0 or severity >= 3:
            return "WARNING"
        else:
            return "INFO"

    def _generate_alert_message(self, region: RiskRegion, 
                                distance: float, depth: float) -> str:
        """生成警报消息"""
        return (f"接近{region.region_type.value}{region.name}，"
                f"距离:{distance:.1f}mm，深度:{depth:.1f}mm，"
                f"严重度:{region.severity}")

    def _get_recommended_action(self, region: RiskRegion,
                               distance: float, depth: float) -> str:
        """获取建议措施"""
        if region.region_type == RiskRegionType.NEURAL_ZONE:
            return "立即停止，侧向偏移>10mm"
        elif region.region_type == RiskRegionType.VASCULAR_ZONE:
            return "紧急停止，回退>5mm，检测出血"
        elif region.region_type == RiskRegionType.JOINT_SPACE:
            return "停止深压，回退至安全深度"
        elif region.region_type == RiskRegionType.ABDOMINAL_ORGAN:
            return "紧急停止，回退至浅层"
        elif region.region_type == RiskRegionType.SPINAL_ZONE:
            return "立即停止，回退至安全距离"
        else:
            return f"保持距离>{region.radius}mm"

    def _get_operational_limit(self, alerts: List[SafetyAlert],
                              distance: float) -> str:
        """获取操作限制"""
        critical = [a for a in alerts if a.alert_level == "CRITICAL"]
        if critical:
            return "禁止继续操作，必须回退"
        
        danger = [a for a in alerts if a.alert_level == "DANGER"]
        if danger:
            return "必须减小深度或偏移"
            
        warning = [a for a in alerts if a.alert_level == "WARNING"]
        if warning:
            return "建议减小深度，保持监控"
            
        if distance < 50:
            return "接近警戒区域，保持警惕"
            
        return "安全，可正常操作"

    def add_risk_region(self, region: RiskRegion):
        """添加风险区域"""
        self._risk_regions.append(region)

    def remove_risk_region(self, name: str) -> bool:
        """移除风险区域"""
        for i, r in enumerate(self._risk_regions):
            if r.name == name:
                del self._risk_regions[i]
                return True
        return False

    def get_alert_history(self, last_n: Optional[int] = None) -> List[SafetyAlert]:
        """获取警报历史"""
        if last_n is None:
            return self._alert_history.copy()
        return self._alert_history[-last_n:]

    def clear_alert_history(self):
        """清空警报历史"""
        self._alert_history = []

    def create_exclusion_zone(self, center: Tuple[float, float, float],
                             radius: float) -> RiskRegion:
        """创建排斥区域 (自动添加为风险区域)"""
        zone = RiskRegion(
            region_type=RiskRegionType.NEURAL_ZONE,  # 使用神经区作为通用
            name=f"排斥区_{center}",
            center=center,
            radius=radius,
            severity=4,
            description="用户定义的避让区域"
        )
        self.add_risk_region(zone)
        return zone


class TrajectorySafetyChecker:
    """轨迹安全检查器
    
    检查机械臂运动轨迹的安全性
    """

    def __init__(self, safety_monitor: SafetyMonitor):
        self._monitor = safety_monitor
        self._waypoints: List[Tuple[float, float, float]] = []

    def add_waypoint(self, position: Tuple[float, float, float]):
        """添加路径点"""
        self._waypoints.append(position)

    def check_trajectory(self, depth: float = 0.0) -> Tuple[bool, List[SafetyAlert]]:
        """检查整条轨迹的安全性
        
        Returns:
            (是否安全, 警报列表)
        """
        all_alerts: List[SafetyAlert] = []
        
        for i, pos in enumerate(self._waypoints):
            status = self._monitor.check_position(pos, depth)
            if not status.is_safe or status.alerts:
                all_alerts.extend(status.alerts)
        
        unsafe_alerts = [a for a in all_alerts if a.alert_level in ('CRITICAL', 'DANGER')]
        return len(unsafe_alerts) == 0, all_alerts

    def suggest_safe_alternatives(self, target: Tuple[float, float, float],
                                  risk_region: RiskRegion) -> List[Tuple[str, Tuple[float, float, float]]]:
        """建议安全替代路径"""
        alternatives = []
        
        # 方案1: 侧向偏移
        offset_x = risk_region.radius + 10
        alternatives.append(
            ("X轴偏移+{:.0f}mm".format(offset_x), 
             (target[0] + offset_x, target[1], target[2]))
        )
        alternatives.append(
            ("X轴偏移-{:.0f}mm".format(offset_x),
             (target[0] - offset_x, target[1], target[2]))
        )
        
        # 方案2: 浅层偏移
        alternatives.append(
            ("深度减小10mm",
             (target[0], target[1], target[2] + 10))
        )
        
        # 过滤有效的替代方案
        valid_alts = []
        for desc, pos in alternatives:
            dist = self._monitor._calculate_distance(pos, risk_region.center)
            if dist > risk_region.radius:
                valid_alts.append((desc, pos))
        
        return valid_alts

    def clear_waypoints(self):
        """清空路径点"""
        self._waypoints = []


def main():
    """测试安全监控器"""
    print("=" * 60)
    print("安全监控器测试")
    print("=" * 60)
    
    # 创建监控器
    monitor = SafetyMonitor()
    
    print(f"\n风险区域数量: {len(monitor._risk_regions)}")
    for r in monitor._risk_regions[:5]:
        print(f"  - {r.name} ({r.region_type.value}): {r.description[:20]}...")
    
    # 测试位置检查
    print("\n" + "=" * 60)
    print("位置安全检查")
    print("=" * 60)
    
    test_positions = [
        ("足三里", (120.0, -45.0, -380.0)),
        ("接近腘动脉", (85.0, -15.0, -345.0)),
        ("肾脏区域", (-60.0, -70.0, -145.0)),
        ("安全区域", (0.0, 0.0, 0.0)),
    ]
    
    for name, pos in test_positions:
        status = monitor.check_position(pos, depth=15.0)
        print(f"\n{name}: {pos}")
        print(f"  安全: {status.is_safe}")
        print(f"  最近风险: {status.nearest_risk_name} ({status.distance_to_nearest_risk:.1f}mm)")
        print(f"  风险等级: {status.overall_risk_level}")
        print(f"  限制: {status.operational_limit}")
        
        if status.alerts:
            print("  警报:")
            for alert in status.alerts:
                print(f"    [{alert.alert_level}] {alert.message}")
                print(f"    建议: {alert.recommended_action}")
    
    # 测试轨迹检查
    print("\n" + "=" * 60)
    print("轨迹安全检查")
    print("=" * 60)
    
    checker = TrajectorySafetyChecker(monitor)
    
    # 模拟从浅到深的轨迹
    for z in [-350, -370, -380, -400, -420]:
        checker.add_waypoint((120.0, -45.0, z))
    
    is_safe, alerts = checker.check_trajectory(depth=20.0)
    print(f"轨迹安全: {is_safe}")
    print(f"警报数量: {len(alerts)}")
    
    for alert in alerts:
        print(f"  [{alert.alert_level}] {alert.message}")
    
    # 测试替代路径建议
    print("\n" + "=" * 60)
    print("安全替代路径建议")
    print("=" * 60)
    
    # 接近犊鼻时的替代
    target_pos = (115.0, -5.0, -370.0)
    knee_risk = monitor._find_region("膝关节腔")
    if knee_risk:
        alts = checker.suggest_safe_alternatives(target_pos, knee_risk)
        for desc, pos in alts:
            print(f"  {desc}: {pos}")


if __name__ == "__main__":
    main()
