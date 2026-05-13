# -*- coding: utf-8 -*-
"""
描述统计脚本

输入：data/panda_reviews_clean.csv（clean_data.py 的产出）
输出：
  - 终端打印数据概览（总数、来源、评分分布、时间分布、字数）
  - figs/score_distribution.png       评分分布柱状图
  - figs/length_distribution.png      字数分布直方图
  - figs/monthly_trend.png            按月评论量趋势

用于报告"数据采集"章节的描述与配图。
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境也能出图
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

# ============== 配置 ==============
CLEAN_CSV = Path("data/panda_reviews_clean.csv")
FIG_DIR = Path("figs")
FIG_DIR.mkdir(exist_ok=True)

# 让 matplotlib 显示中文（Windows 默认有 SimHei / Microsoft YaHei）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def load_rows():
    if not CLEAN_CSV.exists():
        print(f"[error] 找不到 {CLEAN_CSV}，请先跑 clean_data.py")
        sys.exit(1)
    with CLEAN_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def print_overview(rows):
    """终端打印数据描述（写报告时直接抄）"""
    n = len(rows)
    print("=" * 60)
    print(f"总评论数: {n}")
    print()

    # 来源分布
    src_counter = Counter(r.get("fromTypeText", "未知") or "未知" for r in rows)
    print("来源分布:")
    for k, v in src_counter.most_common():
        print(f"  {k:<20} {v} 条 ({v/n*100:.1f}%)")
    print()

    # 评分分布
    score_counter = Counter()
    for r in rows:
        try:
            s = int(float(r.get("score") or 0))
            score_counter[s] += 1
        except ValueError:
            pass
    print("评分分布:")
    for s in sorted(score_counter.keys(), reverse=True):
        cnt = score_counter[s]
        print(f"  {s} 星  {cnt:>5} 条 ({cnt/n*100:.1f}%)  {'█' * int(cnt/n*40)}")
    print()

    # 字数统计
    lengths = [len(r.get("content_clean", "")) for r in rows]
    avg = sum(lengths) / len(lengths)
    print(f"字数: 平均 {avg:.1f}  最长 {max(lengths)}  最短 {min(lengths)}")
    print()

    # 时间范围（取 publishDate 的 YYYY-MM）
    months = [r.get("publishDate", "")[:7] for r in rows if r.get("publishDate", "")[:4].isdigit()]
    months.sort()
    if months:
        print(f"时间范围: {months[0]}  ~  {months[-1]}")
    print("=" * 60)
    return score_counter, lengths, months


def plot_score(score_counter):
    keys = sorted(score_counter.keys())
    vals = [score_counter[k] for k in keys]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar([f"{k}星" for k in keys], vals, color="#FF8C42")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v, str(v), ha="center", va="bottom", fontsize=10)
    ax.set_title("评分分布")
    ax.set_ylabel("评论数")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    out = FIG_DIR / "score_distribution.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig] {out}")


def plot_length(lengths):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    # 截断超长尾巴避免压扁主体
    cap = min(500, max(lengths))
    capped = [min(L, cap) for L in lengths]
    ax.hist(capped, bins=40, color="#4A90E2", edgecolor="white")
    ax.set_title(f"评论字数分布（截断到 {cap} 字以便可视化）")
    ax.set_xlabel("字数")
    ax.set_ylabel("评论数")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    out = FIG_DIR / "length_distribution.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig] {out}")


def plot_monthly(months):
    if not months:
        print("[skip] 无有效时间字段，跳过月度趋势图")
        return
    counter = Counter(months)
    keys = sorted(counter.keys())
    vals = [counter[k] for k in keys]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(keys, vals, marker="o", color="#52B788", linewidth=2)
    ax.fill_between(keys, vals, alpha=0.2, color="#52B788")
    ax.set_title("评论数月度变化")
    ax.set_ylabel("评论数")
    ax.tick_params(axis="x", rotation=45)
    # x 轴太密时只显示部分刻度
    if len(keys) > 18:
        step = len(keys) // 12 + 1
        for i, lab in enumerate(ax.get_xticklabels()):
            if i % step != 0:
                lab.set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    out = FIG_DIR / "monthly_trend.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig] {out}")


def main():
    rows = load_rows()
    score_counter, lengths, months = print_overview(rows)
    plot_score(score_counter)
    plot_length(lengths)
    plot_monthly(months)
    print(f"\n[done] 图表保存到 {FIG_DIR}/")


if __name__ == "__main__":
    main()
