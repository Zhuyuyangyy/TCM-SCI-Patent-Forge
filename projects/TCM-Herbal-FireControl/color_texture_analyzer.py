"""
color_texture_analyzer.py - Processing Color Texture Change Analyzer
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ColorDescriptor:
    hsv_h: float
    hsv_s: float
    hsv_v: float
    rgb_r: int
    rgb_g: int
    rgb_b: int
    lab_l: float
    lab_a: float
    lab_b: float
    color_name: str


@dataclass
class TextureDescriptor:
    roughness: float
    granularity: float
    gloss: float
    uniformity: float
    pattern: str


@dataclass
class ColorTextureResult:
    before_color: ColorDescriptor
    after_color: ColorDescriptor
    before_texture: TextureDescriptor
    after_texture: TextureDescriptor
    color_delta_e: float
    color_change_ratio: float
    texture_change_ratio: float
    quality_indicator: str
    analysis_report: List[str]


class ColorAnalyzer:
    COLOR_NAMES = {
        (0, 15): "Red", (15, 45): "Orange-Yellow", (45, 70): "Yellow",
        (70, 150): "Green", (150, 200): "Cyan", (200, 260): "Blue",
        (260, 290): "Purple", (290, 330): "Pink", (330, 360): "Red"
    }

    def __init__(self):
        self.rgb_reference = {}

    def rgb_to_hsv(self, r: int, g: int, b: int) -> Tuple[float, float, float]:
        r, g, b = r/255.0, g/255.0, b/255.0
        max_c, min_c = max(r, g, b), min(r, g, b)
        diff = max_c - min_c
        if diff == 0: h = 0
        elif max_c == r: h = (60 * ((g - b) / diff) + 360) % 360
        elif max_c == g: h = (60 * ((b - r) / diff) + 120) % 360
        else: h = (60 * ((r - g) / diff) + 240) % 360
        s = 0 if max_c == 0 else diff / max_c
        v = max_c
        return h, s, v

    def rgb_to_lab(self, r: int, g: int, b: int) -> Tuple[float, float, float]:
        r, g, b = r/255.0, g/255.0, b/255.0
        def transform(c): return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        r, g, b = transform(r), transform(g), transform(b)
        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
        x, y, z = x / 0.95047, y / 1.0, z / 1.08883
        def f(t):
            delta = 6/29
            return t ** (1/3) if t > delta ** 3 else t / (3 * delta ** 2) + 4/29
        fx, fy, fz = f(x), f(y), f(z)
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b_lab = 200 * (fy - fz)
        return L, a, b_lab

    def get_color_name(self, h: float, s: float, v: float) -> str:
        for (h_start, h_end), name in self.COLOR_NAMES.items():
            if h_start <= h < h_end:
                if s < 0.1:
                    if v > 0.8: return "Gray-White"
                    elif v < 0.3: return "Black"
                    else: return "Gray"
                return name
        return "Unknown"

    def analyze_color(self, rgb_tuple: Tuple[int, int, int]) -> ColorDescriptor:
        r, g, b = rgb_tuple
        h, s, v = self.rgb_to_hsv(r, g, b)
        L, a, b_lab = self.rgb_to_lab(r, g, b)
        return ColorDescriptor(hsv_h=round(h, 1), hsv_s=round(s, 3), hsv_v=round(v, 3),
            rgb_r=r, rgb_g=g, rgb_b=b, lab_l=round(L, 2), lab_a=round(a, 2), lab_b=round(b_lab, 2),
            color_name=self.get_color_name(h, s, v))

    def calculate_delta_e(self, lab1, lab2) -> float:
        L1, a1, b1 = lab1
        L2, a2, b2 = lab2
        return np.sqrt((L2-L1)**2 + (a2-a1)**2 + (b2-b1)**2)

    def simulate_herb_color(self, herb_name: str, noise_level: float = 0.05):
        from herb_database import get_herb_info
        herb = get_herb_info(herb_name)
        if not herb:
            raw_rgb, processed_rgb = (200, 200, 200), (150, 120, 100)
        else:
            raw_color_str = herb.get("color_change", {}).get("raw", "Gray-White")
            processed_color_str = herb.get("color_change", {}).get("processed", "Yellow-Brown")
            raw_rgb = self._color_name_to_rgb(raw_color_str)
            processed_rgb = self._color_name_to_rgb(processed_color_str)
        raw_rgb = tuple(max(0, min(255, c + int(np.random.randn() * noise_level * 255))) for c in raw_rgb)
        processed_rgb = tuple(max(0, min(255, c + int(np.random.randn() * noise_level * 255))) for c in processed_rgb)
        return self.analyze_color(raw_rgb), self.analyze_color(processed_rgb)

    def _color_name_to_rgb(self, color_name: str) -> Tuple[int, int, int]:
        mapping = {"Gray-White": (220, 218, 210), "Yellow-White": (255, 250, 220),
            "Yellow-Brown": (180, 140, 90), "Red-Brown": (139, 90, 70),
            "Black-Brown": (60, 45, 35), "Char-Brown": (100, 70, 50),
            "Golden": (255, 215, 100), "Deep-Yellow": (220, 180, 60)}
        return mapping.get(color_name, (180, 160, 130))


class TextureAnalyzer:
    def analyze_texture(self, roughness: float, granularity: float, gloss: float, uniformity: float) -> TextureDescriptor:
        if roughness < 0.3 and gloss > 0.5: pattern = "Smooth"
        elif roughness > 0.6: pattern = "Rough"
        elif granularity > 0.5: pattern = "Granular"
        elif gloss > 0.3 and roughness < 0.5: pattern = "Fine"
        elif uniformity < 0.5: pattern = "Uneven"
        else: pattern = "General"
        return TextureDescriptor(roughness=round(roughness, 3), granularity=round(granularity, 3),
            gloss=round(gloss, 3), uniformity=round(uniformity, 3), pattern=pattern)

    def simulate_herb_texture(self, herb_name: str, noise_level: float = 0.1):
        raw_params = {"roughness": 0.6, "granularity": 0.4, "gloss": 0.2, "uniformity": 0.7}
        processed_params = {"roughness": 0.3, "granularity": 0.3, "gloss": 0.5, "uniformity": 0.85}
        raw_params = {k: max(0, min(1, v + np.random.randn() * noise_level)) for k, v in raw_params.items()}
        processed_params = {k: max(0, min(1, v + np.random.randn() * noise_level)) for k, v in processed_params.items()}
        return self.analyze_texture(**raw_params), self.analyze_texture(**processed_params)

    def calculate_texture_similarity(self, tex1, tex2) -> float:
        diff = abs(tex1.roughness - tex2.roughness) + abs(tex1.granularity - tex2.granularity) + abs(tex1.gloss - tex2.gloss) + abs(tex1.uniformity - tex2.uniformity)
        return max(0, 1 - diff / 4)


class ColorTextureAnalyzer:
    def __init__(self):
        self.color_analyzer = ColorAnalyzer()
        self.texture_analyzer = TextureAnalyzer()

    def analyze(self, herb_name: str, processing_method: str, raw_rgb=None, processed_rgb=None) -> ColorTextureResult:
        if raw_rgb and processed_rgb:
            raw_color = self.color_analyzer.analyze_color(raw_rgb)
            processed_color = self.color_analyzer.analyze_color(processed_rgb)
        else:
            raw_color, processed_color = self.color_analyzer.simulate_herb_color(herb_name)
        raw_texture, processed_texture = self.texture_analyzer.simulate_herb_texture(herb_name)
        raw_lab = (raw_color.lab_l, raw_color.lab_a, raw_color.lab_b)
        processed_lab = (processed_color.lab_l, processed_color.lab_a, processed_color.lab_b)
        delta_e = self.color_analyzer.calculate_delta_e(raw_lab, processed_lab)
        raw_hsv = (raw_color.hsv_h, raw_color.hsv_s, raw_color.hsv_v)
        processed_hsv = (processed_color.hsv_h, processed_color.hsv_s, processed_color.hsv_v)
        color_change_ratio = np.sqrt(sum((a-b)**2 for a, b in zip(raw_hsv, processed_hsv))) / np.sqrt(360**2 + 1**2 + 1**2)
        texture_similarity = self.texture_analyzer.calculate_texture_similarity(raw_texture, processed_texture)
        texture_change_ratio = 1 - texture_similarity
        quality_indicator = self._judge_quality(delta_e, color_change_ratio, texture_change_ratio)
        report = self._generate_report(herb_name, processing_method, raw_color, processed_color, raw_texture, processed_texture, delta_e, color_change_ratio, texture_change_ratio, quality_indicator)
        return ColorTextureResult(before_color=raw_color, after_color=processed_color, before_texture=raw_texture, after_texture=processed_texture, color_delta_e=round(delta_e, 2), color_change_ratio=round(color_change_ratio, 3), texture_change_ratio=round(texture_change_ratio, 3), quality_indicator=quality_indicator, analysis_report=report)

    def _judge_quality(self, delta_e: float, color_ratio: float, texture_ratio: float) -> str:
        if delta_e > 25 and texture_ratio > 0.3: return "excellent"
        elif delta_e > 15 and texture_ratio > 0.2: return "good"
        elif delta_e > 8 or texture_ratio > 0.1: return "acceptable"
        else: return "poor"

    def _generate_report(self, herb_name: str, method: str, raw_color, processed_color, raw_texture, processed_texture, delta_e: float, color_change_ratio: float, texture_ratio: float, quality: str) -> List[str]:
        report = [f"=== {herb_name} - {method} Color Texture Analysis ===",
            f"Color: {raw_color.color_name} -> {processed_color.color_name}, Delta E: {delta_e:.2f}",
            f"Texture: {raw_texture.pattern} -> {processed_texture.pattern}",
            f"Quality: {quality.upper()}"]
        return report


def analyze_color_texture(herb_name: str, processing_method: str, raw_rgb=None, processed_rgb=None) -> ColorTextureResult:
    return ColorTextureAnalyzer().analyze(herb_name, processing_method, raw_rgb, processed_rgb)


if __name__ == "__main__":
    result = analyze_color_texture("Aconite", "Processed")
    print(f"Delta E: {result.color_delta_e}, Quality: {result.quality_indicator}")
