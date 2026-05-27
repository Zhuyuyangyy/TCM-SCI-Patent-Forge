"""
demo.py - 附子炮制火候智能判定演示
输入附子 + 目标火候 → 温度曲线 → 火候判定 → 质量评估报告
"""

import sys
import time
from typing import Dict, List, Optional

# 导入项目模块
from herb_database import get_herb_info, get_processing_methods, get_fire_parameters, list_all_herbs
from fire_sensor_sim import generate_fuzhou_processing_curve, generate_custom_herb_curve, FireSensorSimulator
from fire_judge_model import HybridFireJudge, judge_fire, FireJudgmentResult
from color_texture_analyzer import ColorTextureAnalyzer, analyze_color_texture, ColorTextureResult


class FireControlDemo:
    """火候智能判定演示系统"""
    
    def __init__(self):
        self.herb_name = "附子"
        self.processing_method = "制附子"
        self.target_temp = 120
        self.target_duration = 240
        
        # 各模块实例
        self.fire_judge = HybridFireJudge()
        self.color_analyzer = ColorTextureAnalyzer()
        
    def print_header(self, title: str):
        """打印标题"""
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)
    
    def print_section(self, title: str):
        """打印章节标题"""
        print(f"\n--- {title} ---")
    
    def run_full_demo(self, method: str = "制附子", duration: float = 240):
        """
        运行完整演示流程
        
        Args:
            method: 炮制方法
            duration: 持续时间(分钟)
        """
        self.print_header("H05中药炮制火候智能判定系统")
        print(f"药材: {self.herb_name}")
        print(f"炮制方法: {method}")
        
        # 更新参数
        self.processing_method = method
        self.target_duration = duration
        params = get_fire_parameters(self.herb_name, method)
        if params:
            self.target_temp = params.get("temp_target", 120)
        
        # ========== 第一步：获取药材信息 ==========
        self._step_herb_info()
        
        # ========== 第二步：生成温度曲线 ==========
        self._step_temperature_curve()
        
        # ========== 第三步：火候判定 ==========
        fire_result = self._step_fire_judgment()
        
        # ========== 第四步：颜色纹理分析 ==========
        color_result = self._step_color_texture_analysis()
        
        # ========== 第五步：生成综合报告 ==========
        self._generate_quality_report(fire_result, color_result)
        
        return fire_result, color_result
    
    def _step_herb_info(self):
        """步骤1：显示药材信息"""
        self.print_section("步骤1: 药材信息查询")
        
        herb = get_herb_info(self.herb_name)
        if not herb:
            print(f"未找到药材: {self.herb_name}")
            return
        
        print(f"中文名: {herb['name_cn']}")
        print(f"英文名: {herb['name_en']}")
        print(f"类别: {herb['category']}")
        print(f"来源: {herb['origin']}")
        
        print(f"\n可用炮制方法:")
        methods = get_processing_methods(self.herb_name)
        for m in methods:
            params = get_fire_parameters(self.herb_name, m)
            if params:
                print(f"  - {m}: 目标温度{params['temp_target']}℃, 持续{params['duration_min']}分钟")
        
        # 显示当前选择的炮制方法详情
        current_params = get_fire_parameters(self.herb_name, self.processing_method)
        if current_params:
            print(f"\n当前炮制参数:")
            print(f"  目标温度: {current_params['temp_target']}℃")
            print(f"  温度范围: {current_params['temp_range']}℃")
            print(f"  持续时间: {current_params['duration_min']}分钟")
            print(f"  说明: {current_params['description']}")
            
            # 火力等级
            fire_levels = herb.get('fire_level', {})
            print(f"  火力等级: 低火{fire_levels.get('low', 120)}℃ / 中火{fire_levels.get('medium', 180)}℃ / 高火{fire_levels.get('high', 230)}℃")
        
        # 质量标志物
        markers = herb.get('quality_markers', [])
        print(f"  质量标志物: {', '.join(markers)}")
        
        # 颜色纹理变化
        color_change = herb.get('color_change', {})
        print(f"  颜色变化: {color_change.get('raw', 'N/A')} → {color_change.get('processed', 'N/A')}")
        
        texture_change = herb.get('texture_change', {})
        print(f"  质地变化: {texture_change.get('raw', 'N/A')} → {texture_change.get('processed', 'N/A')}")
    
    def _step_temperature_curve(self) -> tuple:
        """步骤2：生成温度曲线"""
        self.print_section("步骤2: 温度曲线生成")
        
        print(f"正在生成 {self.herb_name} - {self.processing_method} 的温度曲线...")
        print(f"目标温度: {self.target_temp}℃")
        print(f"持续时间: {self.target_duration}分钟")
        
        # 生成标准曲线
        times, temps = generate_fuzhou_processing_curve(
            method=self.processing_method,
            duration_min=self.target_duration
        )
        
        # 创建模拟器获取统计
        sim = FireSensorSimulator()
        sim.time_series = times
        sim.temp_series = temps
        
        stats = sim.get_curve_statistics()
        
        print(f"\n曲线生成完成!")
        print(f"  数据点: {len(temps)}")
        print(f"  平均温度: {stats['mean_temp']:.1f}℃")
        print(f"  温度标准差: {stats['std_temp']:.2f}℃")
        print(f"  温度范围: {stats['min_temp']:.1f} - {stats['max_temp']:.1f}℃")
        print(f"  曲线稳定性: {stats['stability']:.2%}")
        
        # 模拟实时采集（显示前10个和后10个点）
        print(f"\n温度数据采样 (前5点):")
        for i in range(5):
            print(f"  t={times[i]:5d}s: {temps[i]:6.1f}℃")
        
        print(f"  ... ({len(temps)-10} 个数据点省略) ...\n")
        
        print(f"温度数据采样 (后5点):")
        for i in range(len(temps)-5, len(temps)):
            print(f"  t={times[i]:5d}s: {temps[i]:6.1f}℃")
        
        self.current_times = times
        self.current_temps = temps
        
        return times, temps
    
    def _step_fire_judgment(self) -> FireJudgmentResult:
        """步骤3：火候判定"""
        self.print_section("步骤3: 火候智能判定")
        
        print("正在进行火候判定...")
        print(f"  药材: {self.herb_name}")
        print(f"  方法: {self.processing_method}")
        print(f"  目标温度: {self.target_temp}℃")
        print(f"  目标时间: {self.target_duration}分钟")
        
        # 调用混合判定系统
        result = judge_fire(
            herb_name=self.herb_name,
            processing_method=self.processing_method,
            actual_curve=self.current_temps,
            time_points=self.current_times,
            target_temp=self.target_temp,
            target_duration=self.target_duration
        )
        
        # 显示判定结果
        print(f"\n===== 火候判定结果 =====")
        print(f"综合等级: {result.fire_level}")
        print(f"综合得分: {result.score}/100")
        print(f"判定置信度: {result.confidence:.2%}")
        print(f"是否合格: {'✓ 是' if result.is_qualified else '✗ 否'}")
        
        print(f"\n温度偏差: {result.temp_deviation:.1f}℃")
        
        if result.reasons:
            print(f"\n问题分析:")
            for reason in result.reasons:
                print(f"  - {reason}")
        
        if result.suggestions:
            print(f"\n改进建议:")
            for sug in result.suggestions:
                print(f"  • {sug}")
        
        # 阶段分析
        print(f"\n各阶段分析:")
        for phase, info in result.phase_analysis.items():
            print(f"  {phase}:")
            print(f"    得分: {info.get('score', 'N/A')}")
            if 'mean_temp' in info:
                print(f"    平均温度: {info['mean_temp']}℃")
            if 'analysis' in info:
                print(f"    分析: {info['analysis']}")
        
        return result
    
    def _step_color_texture_analysis(self) -> ColorTextureResult:
        """步骤4：颜色纹理分析"""
        self.print_section("步骤4: 颜色纹理分析")
        
        print("正在进行颜色纹理分析...")
        
        result = analyze_color_texture(
            herb_name=self.herb_name,
            processing_method=self.processing_method
        )
        
        print(f"\n===== 颜色纹理分析结果 =====")
        print(f"色差ΔE: {result.color_delta_e}")
        print(f"颜色变化率: {result.color_change_ratio*100:.1f}%")
        print(f"纹理变化率: {result.texture_change_ratio*100:.1f}%")
        print(f"质量等级: {result.quality_indicator.upper()}")
        
        print(f"\n炮制前:")
        print(f"  颜色: {result.before_color.color_name} (RGB:{result.before_color.rgb_r},{result.before_color.rgb_g},{result.before_color.rgb_b})")
        print(f"  质地: {result.before_texture.pattern} (粗糙度:{result.before_texture.roughness:.2f}, 光泽度:{result.before_texture.gloss:.2f})")
        
        print(f"\n炮制后:")
        print(f"  颜色: {result.after_color.color_name} (RGB:{result.after_color.rgb_r},{result.after_color.rgb_g},{result.after_color.rgb_b})")
        print(f"  质地: {result.after_texture.pattern} (粗糙度:{result.after_texture.roughness:.2f}, 光泽度:{result.after_texture.gloss:.2f})")
        
        return result
    
    def _generate_quality_report(self, fire_result: FireJudgmentResult, color_result: ColorTextureResult):
        """生成综合质量评估报告"""
        self.print_header("综合质量评估报告")
        
        herb = get_herb_info(self.herb_name)
        
        print(f"药材名称: {self.herb_name} ({herb['name_en'] if herb else 'N/A'})")
        print(f"炮制方法: {self.processing_method}")
        print(f"报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n" + "-" * 50)
        
        # 1. 火候控制评估
        print("\n【一、火候控制评估】")
        fire_score = fire_result.score
        fire_level_desc = {
            "qualified": "优秀",
            "acceptable": "合格",
            "overheat": "过热",
            "insufficient": "不足",
            "failed": "失败"
        }.get(fire_result.fire_level, "未知")
        
        print(f"  判定等级: {fire_level_desc} ({fire_result.fire_level})")
        print(f"  综合得分: {fire_score}/100")
        print(f"  温度控制: {'优秀' if fire_result.temp_deviation < 5 else '良好' if fire_result.temp_deviation < 10 else '一般'}")
        
        # 阶段评分
        phase_scores = []
        for phase, info in fire_result.phase_analysis.items():
            if 'score' in info:
                phase_scores.append(info['score'])
        
        if phase_scores:
            print(f"  各阶段平均得分: {sum(phase_scores)/len(phase_scores):.1f}")
        
        # 2. 外观质量评估
        print("\n【二、外观质量评估】")
        quality_desc = {
            "excellent": "优秀",
            "good": "良好",
            "acceptable": "一般",
            "poor": "较差"
        }.get(color_result.quality_indicator, "未知")
        
        print(f"  颜色纹理等级: {quality_desc} ({color_result.quality_indicator})")
        print(f"  色差ΔE: {color_result.color_delta_e} ({'变化显著' if color_result.color_delta_e > 20 else '变化适中' if color_result.color_delta_e > 10 else '变化不明显'})")
        print(f"  颜色变化: {color_result.before_color.color_name} → {color_result.after_color.color_name}")
        print(f"  质地变化: {color_result.before_texture.pattern} → {color_result.after_texture.pattern}")
        
        # 3. 综合评定
        print("\n【三、综合评定】")
        
        # 计算加权总分
        total_score = fire_score * 0.7 + (100 - min(100, color_result.color_delta_e)) * 0.3
        
        if total_score >= 85 and fire_result.is_qualified:
            final_rating = "优"
            final_desc = "炮制质量优秀，完全符合标准"
        elif total_score >= 70 and fire_result.is_qualified:
            final_rating = "良"
            final_desc = "炮制质量良好，达到合格标准"
        elif total_score >= 60:
            final_rating = "中"
            final_desc = "炮制质量一般，建议优化工艺参数"
        else:
            final_rating = "差"
            final_desc = "炮制质量不达标，需要重新加工"
        
        print(f"  综合评分: {total_score:.1f}/100")
        print(f"  质量等级: {final_rating}")
        print(f"  评定结论: {final_desc}")
        
        # 4. 详细数据表
        print("\n【四、详细数据汇总】")
        print(f"  {'指标':<20} {'数值':<15} {'单位':<10} {'评价'}")
        print(f"  {'-'*60}")
        print(f"  {'目标温度':<20} {self.target_temp:<15} {'℃':<10} {'-'}")
        print(f"  {'目标时间':<20} {self.target_duration:<15} {'分钟':<10} {'-'}")
        print(f"  {'实际平均温度':<20} {fire_result.temp_deviation + self.target_temp:<15.1f} {'℃':<10} {'-'}")
        print(f"  {'温度偏差':<20} {fire_result.temp_deviation:<15.1f} {'℃':<10} {'+' if fire_result.temp_deviation > 0 else ''}{-fire_result.temp_deviation if fire_result.temp_deviation < 0 else ''}")
        print(f"  {'色差ΔE':<20} {color_result.color_delta_e:<15.2f} {'':<10} {'显著' if color_result.color_delta_e > 20 else '适中' if color_result.color_delta_e > 10 else '轻微'}")
        print(f"  {'判定置信度':<20} {fire_result.confidence:<15.2%} {'':<10} {'高' if fire_result.confidence > 0.8 else '中' if fire_result.confidence > 0.5 else '低'}")
        
        # 5. 建议
        print("\n【五、优化建议】")
        if fire_result.suggestions:
            for i, sug in enumerate(fire_result.suggestions, 1):
                print(f"  {i}. {sug}")
        else:
            print("  当前工艺参数控制良好，无需特别调整。")
        
        # 签名
        print("\n" + "-" * 50)
        print("报告结束 - TCM-Herbal-FireControl 火候智能判定系统")
        print("-" * 50)


def interactive_demo():
    """交互式演示"""
    demo = FireControlDemo()
    
    print("\n欢迎使用中药炮制火候智能判定系统")
    print("=" * 50)
    
    # 显示可选药材
    print("\n支持的药材列表:")
    herbs = list_all_herbs()
    for i, herb in enumerate(herbs[:10], 1):
        print(f"  {i}. {herb}")
    if len(herbs) > 10:
        print(f"  ... 共{len(herbs)}种药材")
    
    # 选择药材
    print(f"\n[默认: 附子]")
    herb_choice = input("请选择药材 (直接回车使用默认): ").strip()
    
    if herb_choice:
        demo.herb_name = herb_choice
    
    # 显示该药材的炮制方法
    methods = get_processing_methods(demo.herb_name)
    if not methods:
        print(f"未找到药材: {demo.herb_name}")
        return
    
    print(f"\n{demo.herb_name} 可用的炮制方法:")
    for i, m in enumerate(methods, 1):
        print(f"  {i}. {m}")
    
    # 选择方法
    print(f"\n[默认: {methods[0] if methods else 'N/A'}]")
    method_choice = input("请选择炮制方法 (直接回车使用默认): ").strip()
    
    if method_choice and method_choice.isdigit() and 1 <= int(method_choice) <= len(methods):
        demo.processing_method = methods[int(method_choice) - 1]
    elif method_choice:
        demo.processing_method = method_choice
    
    # 确认并运行
    print(f"\n将运行演示:")
    print(f"  药材: {demo.herb_name}")
    print(f"  方法: {demo.processing_method}")
    
    confirm = input("\n确认运行? (y/n): ").strip().lower()
    if confirm == 'y' or not confirm:
        demo.run_full_demo(demo.processing_method)
    else:
        print("已取消")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="中药炮制火候智能判定演示")
    parser.add_argument("--herb", "-H", default="附子", help="药材名称")
    parser.add_argument("--method", "-M", default="制附子", help="炮制方法")
    parser.add_argument("--duration", "-D", type=float, default=240, help="持续时间(分钟)")
    parser.add_argument("--interactive", "-I", action="store_true", help="交互模式")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_demo()
    else:
        demo = FireControlDemo()
        demo.herb_name = args.herb
        demo.processing_method = args.method
        demo.target_duration = args.duration
        demo.run_full_demo(args.method, args.duration)
