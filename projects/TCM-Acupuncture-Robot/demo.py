"""
TCM-Acupuncture-Robot E01
机械臂推拿力度-穴位-形变闭环控制
 demo.py - 完整演示: 足三里穴位推拿

演示流程:
1. 选择足三里穴位
2. 机械臂定位
3. 12N目标压力PID控制
4. 组织形变计算
5. 安全监控报告
"""

import time
import sys
from typing import Optional

# 导入项目模块
from acupoint_locator import AcupointDatabase, Acupoint
from pressure_controller import PressureController, PIDConfig, AdaptivePressureController
from tissue_deformation import HookeLawModel, MultiLayerViscoelasticModel, TissueType
from safety_monitor import SafetyMonitor, SafetyStatus


class AcupunctureRobotController:
    """针灸推拿机器人控制器"""
    
    def __init__(self):
        print("=" * 70)
        print("TCM-Acupuncture-Robot E01")
        print("机械臂推拿力度-穴位-形变闭环控制系统")
        print("=" * 70)
        
        # 初始化各子系统
        self.acupoint_db = AcupointDatabase()
        self.pressure_ctrl = PressureController(PIDConfig(kp=0.8, ki=0.1, kd=0.3))
        self.adaptive_ctrl = AdaptivePressureController()
        self.tissue_model = HookeLawModel()
        self.visco_model = MultiLayerViscoelasticModel()
        self.safety_monitor = SafetyMonitor()
        
        # 状态变量
        self.current_acupoint: Optional[Acupoint] = None
        self.current_position = (0.0, 0.0, 0.0)
        self.current_depth = 0.0
        self.current_pressure = 0.0
        self.target_pressure = 12.0  # 12N
        
    def select_acupoint(self, name: str) -> bool:
        """选择穴位"""
        acupoint = self.acupoint_db.get_acupoint(name)
        if acupoint is None:
            print(f"[错误] 未找到穴位: {name}")
            return False
            
        self.current_acupoint = acupoint
        self.current_position = acupoint.coordinates
        print(f"\n[穴位选择] {acupoint.name} ({acupoint.name_en})")
        print(f"  经络: {acupoint.channel}")
        print(f"  位置: {acupoint.location}")
        print(f"  坐标: {acupoint.coordinates}")
        print(f"  安全深度: {acupoint.depth[0]} - {acupoint.depth[1]} mm")
        print(f"  风险等级: {acupoint.risk_level}")
        print(f"  禁忌: {acupoint.contraindication}")
        return True
        
    def locate_and_prepare(self) -> bool:
        """定位机械臂到目标位置"""
        if self.current_acupoint is None:
            print("[错误] 未选择穴位")
            return False
            
        print(f"\n[机械臂定位]")
        print(f"  目标位置: {self.current_position}")
        
        # 模拟定位过程
        print("  正在移动到目标位置...")
        time.sleep(0.5)
        
        # 安全检查
        status = self.safety_monitor.check_position(
            self.current_position, 
            depth=0.0
        )
        
        if not status.is_safe:
            print(f"  [警告] 位置存在风险: {status.operational_limit}")
            for alert in status.alerts:
                print(f"    [{alert.alert_level}] {alert.message}")
            if status.overall_risk_level >= 4:
                print("  [错误] 风险过高，取消操作")
                return False
        else:
            print(f"  [安全检查通过] {status.operational_limit}")
            
        print("  定位完成!")
        return True
        
    def execute_tuina(self, duration: float = 5.0) -> dict:
        """执行推拿操作
        
        Args:
            duration: 推拿持续时间 (秒)
            
        Returns:
            执行报告
        """
        if self.current_acupoint is None:
            return {"success": False, "error": "未选择穴位"}
            
        print(f"\n[开始推拿] 目标压力: {self.target_pressure}N, 持续: {duration}s")
        print("-" * 70)
        
        # 初始化控制器
        self.pressure_ctrl.reset()
        self.current_pressure = 0.0
        self.current_depth = 0.0
        
        # 数据记录
        history = []
        start_time = time.time()
        step = 0
        
        # 推拿循环
        while time.time() - start_time < duration:
            step += 1
            timestamp = time.time() - start_time
            
            # 1. PID控制计算
            control_output = self.pressure_ctrl.compute(
                self.target_pressure,
                self.current_pressure,
                timestamp
            )
            
            # 2. 模拟压力响应 (被控对象)
            # 简化的被控对象模型: P(s) = K / (Ts + 1)
            K_object = 0.15  # 过程增益
            T_object = 0.3   # 时间常数
            dt = 0.05
            pressure_delta = (K_object * control_output - self.current_pressure) / T_object * dt
            self.current_pressure += pressure_delta * dt * 10  # 简化
            
            # 添加一些扰动和噪声
            import random
            self.current_pressure += random.gauss(0, 0.2)
            self.current_pressure = max(0, min(20, self.current_pressure))
            
            # 3. 更新深度 (根据压力和组织特性)
            result = self.tissue_model.calculate_deformation_by_depth(
                self.current_pressure,
                self.current_depth
            )
            self.current_depth = result.total_deformation
            
            # 4. 安全监控
            safety_status = self.safety_monitor.check_position(
                self.current_position,
                depth=self.current_depth
            )
            
            # 5. 记录数据
            history.append({
                'time': timestamp,
                'target': self.target_pressure,
                'current': self.current_pressure,
                'error': self.target_pressure - self.current_pressure,
                'depth': self.current_depth,
                'output': control_output,
                'safety': safety_status.is_safe,
                'stiffness': result.effective_stiffness,
                'safety_margin': result.safety_margin,
                'warning': result.warning_level
            })
            
            # 每20步打印一次状态
            if step % 20 == 0:
                state = self.pressure_ctrl.get_history()[-1]
                print(f"  t={timestamp:5.2f}s | 压力: {self.current_pressure:5.2f}N | "
                      f"深度: {self.current_depth:5.2f}mm | 刚度: {result.effective_stiffness:6.2f} | "
                      f"[{result.warning_level}]")
                
                if not safety_status.is_safe:
                    for alert in safety_status.alerts[:1]:
                        print(f"    ⚠️ {alert.alert_level}: {alert.message[:50]}...")
            
            # 检查是否需要停止
            if not safety_status.is_safe:
                critical = [a for a in safety_status.alerts 
                           if a.alert_level in ('CRITICAL', 'DANGER')]
                if critical:
                    print(f"\n[紧急停止] 检测到危险: {critical[0].message}")
                    break
                    
            time.sleep(0.05)  # 50ms控制周期
            
        print("-" * 70)
        print("[推拿完成]")
        
        return {
            'success': True,
            'duration': time.time() - start_time,
            'steps': step,
            'history': history,
            'final_state': history[-1] if history else None
        }
        
    def calculate_deformation_analysis(self, report: dict) -> dict:
        """计算形变分析报告"""
        if not report.get('success') or not report.get('history'):
            return {}
            
        history = report['history']
        
        # 统计计算
        pressures = [h['current'] for h in history]
        depths = [h['depth'] for h in history]
        errors = [h['error'] for h in history]
        margins = [h['safety_margin'] for h in history]
        
        # 压力控制性能
        avg_pressure = sum(pressures) / len(pressures)
        max_pressure = max(pressures)
        min_pressure = min(pressures)
        steady_state_error = sum(errors[-50:]) / len(errors[-50:]) if len(errors) >= 50 else sum(errors) / len(errors)
        
        # 形变统计
        avg_depth = sum(depths) / len(depths)
        max_depth = max(depths)
        
        # 安全统计
        min_safety_margin = min(margins)
        danger_count = sum(1 for h in history if h['warning'] in ('DANGER', 'CRITICAL'))
        
        # 形变分析
        deformation_result = self.tissue_model.calculate_deformation(self.target_pressure)
        
        print("\n" + "=" * 70)
        print("形变分析报告")
        print("=" * 70)
        
        print("\n【1. 压力控制性能】")
        print(f"  目标压力: {self.target_pressure}N")
        print(f"  平均压力: {avg_pressure:.3f}N")
        print(f"  最大压力: {max_pressure:.3f}N")
        print(f"  最小压力: {min_pressure:.3f}N")
        print(f"  稳态误差: {steady_state_error:.3f}N")
        print(f"  控制精度: {(1 - abs(steady_state_error)/self.target_pressure)*100:.1f}%")
        
        print("\n【2. 组织形变统计】")
        print(f"  平均形变深度: {avg_depth:.2f}mm")
        print(f"  最大形变深度: {max_depth:.2f}mm")
        print(f"  等效刚度: {deformation_result.effective_stiffness:.3f} N/mm")
        print(f"  各层形变:")
        for layer, deform in deformation_result.layer_deformations.items():
            print(f"    {layer}: {deform:.3f}mm")
            
        print("\n【3. 安全评估】")
        print(f"  最小安全裕度: {min_safety_margin:.3f}")
        print(f"  危险时刻数: {danger_count}/{len(history)}")
        safety_rating = "优秀" if min_safety_margin > 0.5 else ("良好" if min_safety_margin > 0.3 else ("一般" if min_safety_margin > 0.1 else "危险"))
        print(f"  安全评级: {safety_rating}")
        
        # 胡克定律验证
        print("\n【4. 胡克定律验证 F = k × Δx】")
        theoretical_force = deformation_result.effective_stiffness * deformation_result.total_deformation
        print(f"  理论计算力: {theoretical_force:.3f}N (刚度 × 形变)")
        print(f"  目标作用力: {self.target_pressure:.3f}N")
        print(f"  误差: {abs(theoretical_force - self.target_pressure):.3f}N")
        
        return {
            'avg_pressure': avg_pressure,
            'max_pressure': max_pressure,
            'steady_state_error': steady_state_error,
            'avg_depth': avg_depth,
            'max_depth': max_depth,
            'effective_stiffness': deformation_result.effective_stiffness,
            'min_safety_margin': min_safety_margin,
            'safety_rating': safety_rating,
            'deformation_result': deformation_result
        }
        
    def generate_safety_report(self, report: dict) -> dict:
        """生成安全监控报告"""
        if not report.get('success') or not report.get('history'):
            return {}
            
        history = report['history']
        
        # 统计安全数据
        safety_violations = [h for h in history if not h['safety']]
        warning_events = [h for h in history if h['warning'] in ('WARNING', 'DANGER', 'CRITICAL')]
        
        # 获取所有警报
        all_alerts = self.safety_monitor.get_alert_history()
        
        print("\n" + "=" * 70)
        print("安全监控报告")
        print("=" * 70)
        
        print("\n【1. 风险区域避让】")
        if self.current_acupoint:
            print(f"  当前穴位: {self.current_acupoint.name}")
            print(f"  坐标: {self.current_acupoint.coordinates}")
            
            # 检查附近风险区域
            nearest = self.acupoint_db.find_nearest(self.current_acupoint.coordinates, max_distance=100)
            if nearest:
                ap, dist = nearest
                print(f"  最近穴位: {ap.name} (距离: {dist:.1f}mm)")
        
        print("\n【2. 安全事件统计】")
        print(f"  总采样点: {len(history)}")
        print(f"  安全违规: {len(safety_violations)}")
        print(f"  警告事件: {len(warning_events)}")
        
        # 时间分布
        if warning_events:
            print(f"\n【3. 警告时间分布】")
            early = sum(1 for h in warning_events if h['time'] < 1.0)
            mid = sum(1 for h in warning_events if 1.0 <= h['time'] < 3.0)
            late = sum(1 for h in warning_events if h['time'] >= 3.0)
            print(f"  初期 (0-1s): {early}次")
            print(f"  中期 (1-3s): {mid}次")
            print(f"  后期 (>3s): {late}次")
            
        print("\n【4. 历史警报】")
        if all_alerts:
            for alert in all_alerts[-5:]:  # 最近5条
                print(f"  [{alert.alert_level}] t={alert.timestamp:.1f}s: {alert.message[:40]}...")
        else:
            print("  无历史警报")
            
        print("\n【5. 安全建议】")
        if len(safety_violations) == 0:
            print("  ✅ 整个操作过程安全，无违规事件")
        else:
            print(f"  ⚠️ 检测到 {len(safety_violations)} 次安全违规")
            print("  建议:")
            print("    1. 减小目标压力")
            print("    2. 增加安全裕度")
            print("    3. 优化PID参数")
            
        if len(warning_events) > len(history) * 0.2:
            print("    4. 考虑更换穴位或调整方案")
            
        return {
            'total_samples': len(history),
            'safety_violations': len(safety_violations),
            'warning_events': len(warning_events),
            'all_alerts': all_alerts
        }
        
    def run_full_demo(self):
        """运行完整演示"""
        print("\n" + "=" * 70)
        print("完整演示: 足三里穴位推拿")
        print("=" * 70)
        
        # Step 1: 选择足三里
        print("\n>>> Step 1: 选择穴位")
        if not self.select_acupoint("足三里"):
            return
            
        # 显示穴位列表供选择
        print("\n其他可用穴位:")
        for ap in self.acupoint_db.list_all_acupoints()[:10]:
            print(f"  - {ap.name}")
            
        # Step 2: 定位
        print("\n>>> Step 2: 机械臂定位")
        if not self.locate_and_prepare():
            return
            
        # Step 3: PID压力控制推拿
        print("\n>>> Step 3: PID闭环压力控制 (目标: 12N)")
        report = self.execute_tuina(duration=5.0)
        
        if not report.get('success'):
            print(f"[错误] 推拿执行失败: {report.get('error')}")
            return
            
        # Step 4: 形变计算
        print("\n>>> Step 4: 组织形变计算")
        deform_report = self.calculate_deformation_analysis(report)
        
        # Step 5: 安全报告
        print("\n>>> Step 5: 安全监控报告")
        safety_report = self.generate_safety_report(report)
        
        # 汇总
        print("\n" + "=" * 70)
        print("演示完成汇总")
        print("=" * 70)
        print(f"  穴位: {self.current_acupoint.name if self.current_acupoint else 'N/A'}")
        print(f"  目标压力: {self.target_pressure}N")
        print(f"  实际平均压力: {deform_report.get('avg_pressure', 0):.2f}N")
        print(f"  稳态误差: {deform_report.get('steady_state_error', 0):.2f}N")
        print(f"  最大形变: {deform_report.get('max_depth', 0):.2f}mm")
        print(f"  安全评级: {deform_report.get('safety_rating', 'N/A')}")
        print(f"  安全违规: {safety_report.get('safety_violations', 0)}次")
        print("=" * 70)


def main():
    """主函数"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "TCM-Acupuncture-Robot E01 系统演示" + " " * 20 + "║")
    print("║" + " " * 10 + "机械臂推拿力度-穴位-形变闭环控制" + " " * 22 + "║")
    print("╠" + "═" * 68 + "╣")
    print("║" + " " * 68 + "║")
    print("║  模块列表:                                                         ║")
    print("║    1. acupoint_locator.py     - 20+穴位坐标库                      ║")
    print("║    2. pressure_controller.py  - PID闭环压力控制器                   ║")
    print("║    3. tissue_deformation.py   - 胡克定律弹性模型                   ║")
    print("║    4. safety_monitor.py        - 风险区域避让                       ║")
    print("║    5. demo.py                  - 完整演示程序                       ║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # 创建控制器并运行演示
    controller = AcupunctureRobotController()
    controller.run_full_demo()


if __name__ == "__main__":
    main()
