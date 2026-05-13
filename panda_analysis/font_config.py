# -*- coding: utf-8 -*-
"""
中文字体统一配置模块
======================
作用:
    matplotlib 默认字体不带中文,直接画图会显示成方块(俗称"豆腐块")。
    本模块自动检测系统中可用的中文字体,并提供 set_chinese_font() 函数,
    其他脚本只要在开头 import 它并调用一次,就不用再操心字体问题。

用法(在其他脚本顶部):
    from font_config import set_chinese_font
    set_chinese_font()

直接运行本文件可生成一张测试图,验证字体是否正常:
    python font_config.py
"""

import os
import platform
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager


# 不同系统上常见的中文字体优先级列表(按推荐程度排列)
# 程序会按顺序去找,找到第一个就用
WINDOWS_FONTS = ["Microsoft YaHei", "SimHei", "SimSun", "KaiTi"]
MAC_FONTS = ["PingFang SC", "Heiti TC", "STHeiti", "Arial Unicode MS"]
LINUX_FONTS = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Source Han Sans CN"]


# wordcloud / pillow 需要的是字体文件绝对路径(不能用字体名),
# 与 matplotlib 的 font.sans-serif 机制不同,因此单独维护候选路径表。
_WINDOWS_FONT_FILES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\STKAITI.TTF",
]
_MAC_FONT_FILES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/Library/Fonts/Songti.ttc",
]
_LINUX_FONT_FILES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]


def get_wordcloud_font_path() -> str:
    """返回一个适合 wordcloud/pillow 使用的中文字体文件绝对路径。

    使用者:阶段 1 词云、阶段 4 分组主题词云。
    原先这个逻辑在 01_word_freq.py 和 04_themes.py 分别实现了一份(其中
    04 的简化版只找 Windows 字体,Linux/macOS 会拿到 None)。现统一放在
    本模块,避免漂移。

    找不到时抛 FileNotFoundError。
    """
    sysname = platform.system()
    if sysname == "Windows":
        candidates = _WINDOWS_FONT_FILES
    elif sysname == "Darwin":
        candidates = _MAC_FONT_FILES
    else:
        candidates = _LINUX_FONT_FILES

    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "未找到可用中文字体文件。Windows 请确认 C:\\Windows\\Fonts 下有 "
        "msyh.ttc 或 simhei.ttf;macOS/Linux 请安装 Noto CJK 或思源黑体。"
    )


def list_available_chinese_fonts():
    """
    列出系统里所有 matplotlib 能识别到的、名称里包含中文相关关键词的字体。
    返回一个 list,例如 ['Microsoft YaHei', 'SimHei', ...]
    """
    # 取出 matplotlib 所识别到的全部字体名称
    all_fonts = {f.name for f in font_manager.fontManager.ttflist}
    # 候选关键词:常用中文字体的英文名片段
    keywords = ["YaHei", "SimHei", "SimSun", "KaiTi", "FangSong",
                "PingFang", "Heiti", "STHeiti", "Songti", "Hiragino",
                "Noto Sans CJK", "Noto Serif CJK", "WenQuanYi",
                "Source Han", "Microsoft JhengHei", "Arial Unicode"]
    found = [name for name in all_fonts if any(k.lower() in name.lower() for k in keywords)]
    return sorted(found)


def set_chinese_font(verbose: bool = True) -> str:
    """
    设置 matplotlib 使用中文字体,并解决负号显示成方块的问题。
    参数:
        verbose: 是否打印当前所选字体名(默认 True)
    返回:
        实际启用的字体名;若没找到任何中文字体则返回空字符串
    """
    # 1. 根据操作系统挑选优先字体表
    sysname = platform.system()
    if sysname == "Windows":
        candidates = WINDOWS_FONTS
    elif sysname == "Darwin":   # macOS 内核名 Darwin
        candidates = MAC_FONTS
    else:
        candidates = LINUX_FONTS

    # 2. 把系统已安装的字体名收集成集合,加速比对
    installed = {f.name for f in font_manager.fontManager.ttflist}

    # 3. 按优先级找,找到第一个存在的就用
    chosen = ""
    for font in candidates:
        if font in installed:
            chosen = font
            break

    # 4. 如果优先表里没有,再退而求其次:扫描所有"看起来是中文"的字体
    if not chosen:
        scan = list_available_chinese_fonts()
        if scan:
            chosen = scan[0]

    # 5. 应用到 matplotlib 全局配置
    if chosen:
        # rcParams["font.sans-serif"] 是一个候选列表,放第一个就生效
        matplotlib.rcParams["font.sans-serif"] = [chosen]
        # 解决保存图像时负号 '-' 显示为方块的问题
        matplotlib.rcParams["axes.unicode_minus"] = False
        if verbose:
            print(f"[font_config] 已启用中文字体: {chosen}")
    else:
        if verbose:
            print("[font_config] 警告:未检测到中文字体,图中中文可能显示为方块。")
            print("[font_config] Windows 用户请确认装有微软雅黑(Microsoft YaHei)或黑体(SimHei)。")

    return chosen


def _self_test():
    """
    自测:画一张含中文的小图,保存到 output/中文字体测试.png
    用以肉眼确认字体是否正常显示。
    """
    chosen = set_chinese_font(verbose=True)

    # 列出所有检测到的中文字体,便于排查
    all_cn = list_available_chinese_fonts()
    print(f"[font_config] 检测到 {len(all_cn)} 个中文相关字体:")
    for name in all_cn[:20]:
        print(f"   - {name}")
    if len(all_cn) > 20:
        print(f"   ... 还有 {len(all_cn)-20} 个未列出")

    # 确保输出目录存在
    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", "中文字体测试.png")

    # 画一个简单柱状图,标题/坐标轴/图例都用中文
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["国内游客", "国际游客", "亲子家庭", "情侣夫妻", "朋友出游"]
    values = [1000, 1000, 600, 400, 500]
    ax.bar(categories, values, color=["#3a86ff", "#ffbe0b", "#fb5607", "#ff006e", "#8338ec"])
    ax.set_title("成都大熊猫基地评论分析 — 中文字体测试", fontsize=14)
    ax.set_xlabel("游客分组")
    ax.set_ylabel("评论条数")
    for i, v in enumerate(values):
        ax.text(i, v + 10, str(v), ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[font_config] 测试图已生成: {out_path}")
    print("[font_config] 请打开该图,确认中文显示正常(应该看到完整汉字而不是方块)。")


if __name__ == "__main__":
    _self_test()
