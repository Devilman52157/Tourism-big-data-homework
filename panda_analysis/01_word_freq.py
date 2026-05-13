# -*- coding: utf-8 -*-
"""
01_word_freq.py
================
阶段1:词频分析 + 词云图

产出三类分析:
    1) 整体(2000条)词频 + 词云
    2) 分组对比(国内 vs 国际)词频 + 词云 + 金字塔对比图
    3) 差异词分析(只在一组高频的"特征词")+ 词性分布饼图

用法:
    python 01_word_freq.py            # 全量(2000条)
    python 01_word_freq.py --trial    # 试跑前100条,只输出 Top20 + 一张词云
"""

import argparse
import random
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from wordcloud import WordCloud

# 共用工具:分词、词频、数据加载 —— 见 text_utils.py
from text_utils import (
    load_stopwords, init_jieba, load_data,
    tokenize_dataframe, count_words,
)
# 字体配置模块(阶段0)
from font_config import set_chinese_font, get_wordcloud_font_path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


# ============================================================
# 一、配置常量
# ============================================================
DATA_PATH = "data/panda_sample_2000.csv"
CUSTOM_DICT = "custom_dict.txt"
STOPWORDS_PATH = "stopwords.txt"
OUT_DIR = Path("output/1_word_freq")

# 配色
DOM_COLOR = "#E74C3C"     # 国内游客主色——红
INTL_COLOR = "#1ABC9C"    # 国际游客主色——蓝绿
PANDA_GREEN = "#7CB342"   # 整体词云中点缀的竹叶绿
NEUTRAL_COLOR = "#2C3E50" # 整体条形图主色


# ============================================================
# 二、工具函数(仅本文件独有;分词/词频等见 text_utils)
# ============================================================
def save_freq_csv(counter: Counter, path: Path, top_n: int | None = None) -> None:
    """
    保存词频 CSV。列:rank / word / count / ratio
    ratio = count / 总词数,百分比格式 "x.xx%"
    使用 utf-8-sig,Excel 直接打开不乱码。
    """
    total = sum(counter.values())
    items = counter.most_common(top_n) if top_n else counter.most_common()
    rows = [
        {"rank": i, "word": w, "count": n, "ratio": f"{(n / total * 100):.2f}%"}
        for i, (w, n) in enumerate(items, 1)
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def find_distinctive_words(c_dom: Counter, c_intl: Counter,
                           top_n: int = 200, smooth: int = 5):
    """
    找两组的特征词。
    候选集:出现在国内 Top200 或国际 Top200 的词的并集。
    打分:
        国内特征比 = 国内频次 / (国际频次 + 5)
        国际特征比 = 国际频次 / (国内频次 + 5)
    返回:(国内 Top30 df, 国际 Top30 df)
    """
    top_dom_words = {w for w, _ in c_dom.most_common(top_n)}
    top_intl_words = {w for w, _ in c_intl.most_common(top_n)}
    candidates = top_dom_words | top_intl_words

    rank_dom = {w: i for i, (w, _) in enumerate(c_dom.most_common(), 1)}
    rank_intl = {w: i for i, (w, _) in enumerate(c_intl.most_common(), 1)}

    rows = []
    for w in candidates:
        d = c_dom.get(w, 0)
        ii = c_intl.get(w, 0)
        rows.append({
            "word": w,
            "国内频次": d,
            "国际频次": ii,
            "国内排名": rank_dom.get(w, ""),
            "国际排名": rank_intl.get(w, ""),
            "_dom_score": d / (ii + smooth),
            "_intl_score": ii / (d + smooth),
        })
    df_all = pd.DataFrame(rows)

    df_dom = (df_all.sort_values("_dom_score", ascending=False)
                    .head(30)
                    .reset_index(drop=True))
    df_intl = (df_all.sort_values("_intl_score", ascending=False)
                     .head(30)
                     .reset_index(drop=True))

    df_dom_out = df_dom.assign(比值=df_dom["_dom_score"].round(3))[
        ["word", "国内频次", "国际频次", "比值", "国内排名", "国际排名"]
    ]
    df_intl_out = df_intl.assign(比值=df_intl["_intl_score"].round(3))[
        ["word", "国内频次", "国际频次", "比值", "国内排名", "国际排名"]
    ]
    return df_dom_out, df_intl_out


# ============================================================
# 五、绘图
# ============================================================
def plot_bar(counter: Counter, top_n: int, title: str, save_path: Path,
             base_color: str) -> None:
    """横向条形图,频次高的颜色深(从浅灰渐变到 base_color)。"""
    items = counter.most_common(top_n)
    # 反向:matplotlib barh 是从下往上画,反向后最高频在顶端
    words = [x[0] for x in items][::-1]
    counts = [x[1] for x in items][::-1]
    n = len(words)

    cmap = mcolors.LinearSegmentedColormap.from_list("grad", ["#dddddd", base_color])
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    fig_h = max(10, n * 0.28)  # 根据词数动态调整
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.barh(words, counts, color=colors)
    ax.set_title(title, fontsize=15, pad=14)
    ax.set_xlabel("频次", fontsize=11)
    # 在每个条形末尾标注数值
    for i, c in enumerate(counts):
        ax.text(c, i, f" {c}", va="center", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _panda_color_func(word, font_size, position, orientation,
                      random_state=None, **kwargs):
    """
    词云配色函数:黑白灰为主、~10% 概率出竹叶绿。
    词云的 color_func 接口固定为这些参数。
    """
    rng = random_state or random
    if rng.random() < 0.10:
        return PANDA_GREEN
    v = rng.randint(20, 110)  # 深灰到中灰
    return f"rgb({v},{v},{v})"


def plot_wordcloud(counter: Counter, save_path: str, font_path: str,
                   palette: str | None = None, colormap: str | None = None,
                   max_words: int = 200) -> None:
    """
    生成词云。
    palette='panda' 时用熊猫主题色函数(黑白灰+少量竹叶绿);
    否则使用传入的 colormap 名(如 'Reds' / 'BuGn')。
    两者都不传时,使用 wordcloud 默认配色。
    """
    wc = WordCloud(
        font_path=font_path,
        width=1920,
        height=1080,
        background_color="white",
        max_words=max_words,
        min_font_size=12,
        max_font_size=240,
        prefer_horizontal=0.9,   # 90% 横向,看起来更整齐
        random_state=42,
        collocations=False,      # 关闭自动二元搭配,避免重复
        margin=4,
    )
    wc.generate_from_frequencies(dict(counter))

    if palette == "panda":
        wc = wc.recolor(color_func=_panda_color_func, random_state=42)
    elif colormap:
        wc = wc.recolor(colormap=colormap, random_state=42)

    wc.to_file(save_path)


def plot_comparison(c_dom: Counter, c_intl: Counter, top_n: int,
                    save_path: Path) -> None:
    """
    国内 vs 国际 金字塔对比图。
    取两组 Top_n 的并集作为共享纵轴,按"国内+国际"频次和升序(让最大值在顶端)。
    左侧:国际(向左,蓝绿) 右侧:国内(向右,红)
    """
    top_dom = [w for w, _ in c_dom.most_common(top_n)]
    top_intl = [w for w, _ in c_intl.most_common(top_n)]
    union = list(dict.fromkeys(top_dom + top_intl))  # 保序并集

    pairs = [(w, c_dom.get(w, 0), c_intl.get(w, 0)) for w in union]
    # 按总和升序:matplotlib barh 自下而上,升序排后最大在顶
    pairs.sort(key=lambda x: x[1] + x[2])

    words = [p[0] for p in pairs]
    dom_vals = [p[1] for p in pairs]
    intl_vals_neg = [-p[2] for p in pairs]   # 国际值取负,使其向左延伸

    fig_h = max(12, len(words) * 0.32)
    fig, ax = plt.subplots(figsize=(15, fig_h))
    ax.barh(words, intl_vals_neg, color=INTL_COLOR, label="国际游客", alpha=0.9)
    ax.barh(words, dom_vals, color=DOM_COLOR, label="国内游客", alpha=0.9)
    ax.axvline(0, color="black", linewidth=0.8)

    # 数字标注
    for i, (d, neg_i) in enumerate(zip(dom_vals, intl_vals_neg)):
        if d > 0:
            ax.text(d, i, f" {d}", va="center", fontsize=8)
        if neg_i < 0:
            ax.text(neg_i, i, f"{-neg_i} ", va="center", ha="right", fontsize=8)

    # x 轴显示绝对值(隐藏负号)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: str(int(abs(x)))))
    ax.set_title(f"国内游客 vs 国际游客 评论高频词对比(并集 Top{top_n})",
                 fontsize=15, pad=14)
    ax.set_xlabel("频次")
    ax.legend(loc="lower right", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_distinctive(df_dom: pd.DataFrame, df_intl: pd.DataFrame,
                     save_path: Path) -> None:
    """
    上下两子图,各画一组的"特征词 Top20"。
    每个子图:本组频次(主色) + 对照组频次(灰色)叠在一起,直观看出"差异"。
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 16))

    # ---- 上:国内 Top20 ----
    d20 = df_dom.head(20).iloc[::-1]   # 反向使最高排名在顶
    axes[0].barh(d20["word"], d20["国内频次"], color=DOM_COLOR, label="国内频次")
    axes[0].barh(d20["word"], d20["国际频次"], color="#bbbbbb",
                 alpha=0.7, label="国际频次(对照)")
    axes[0].set_title("国内游客最具特色的Top20词", fontsize=14)
    axes[0].set_xlabel("频次")
    axes[0].legend(loc="lower right")
    for i, (d, ii) in enumerate(zip(d20["国内频次"].tolist(),
                                    d20["国际频次"].tolist())):
        axes[0].text(d, i, f"  国内{d} | 国际{ii}", va="center", fontsize=8)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # ---- 下:国际 Top20 ----
    i20 = df_intl.head(20).iloc[::-1]
    axes[1].barh(i20["word"], i20["国际频次"], color=INTL_COLOR, label="国际频次")
    axes[1].barh(i20["word"], i20["国内频次"], color="#bbbbbb",
                 alpha=0.7, label="国内频次(对照)")
    axes[1].set_title("国际游客最具特色的Top20词", fontsize=14)
    axes[1].set_xlabel("频次")
    axes[1].legend(loc="lower right")
    for i, (d, ii) in enumerate(zip(i20["国内频次"].tolist(),
                                    i20["国际频次"].tolist())):
        axes[1].text(ii, i, f"  国际{ii} | 国内{d}", va="center", fontsize=8)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.suptitle("国内 vs 国际游客的特征词对比", fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pos_pie(df: pd.DataFrame, top_n: int, save_path: Path) -> None:
    """
    高频词 Top_n 的词性分布饼图(按种类数,而非频次加权)。
    每个词取它在所有出现中"最频繁的词性"作为代表词性。
    """
    counter = count_words(df["words"])
    top_words = {w for w, _ in counter.most_common(top_n)}

    word_pos_count: dict[str, Counter] = {}
    for pairs in df["pos_pairs"]:
        for w, p in pairs:
            if w in top_words:
                word_pos_count.setdefault(w, Counter())[p] += 1

    cat = Counter()
    for w, pc in word_pos_count.items():
        dominant = pc.most_common(1)[0][0]
        if dominant.startswith("n"):
            cat["名词"] += 1
        elif dominant == "v":
            cat["动词"] += 1
        elif dominant == "a":
            cat["形容词"] += 1
        else:
            cat["其他"] += 1

    labels = list(cat.keys())
    sizes = list(cat.values())
    colors_map = {"名词": "#3498DB", "动词": "#E74C3C",
                  "形容词": "#F1C40F", "其他": "#95A5A6"}
    colors = [colors_map.get(l, "#95A5A6") for l in labels]

    fig, ax = plt.subplots(figsize=(8, 8))
    total = sum(sizes)
    ax.pie(sizes, labels=labels,
           autopct=lambda p: f"{p:.1f}%\n({int(round(p * total / 100))}个)",
           colors=colors, startangle=90,
           textprops={"fontsize": 12},
           wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax.set_title(f"高频词Top{top_n}词性分布(按种类数)", fontsize=14, pad=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 六、主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="阶段1:词频与词云分析")
    parser.add_argument("--trial", action="store_true",
                        help="试跑模式:只用前100条,只输出 Top20 + 一张词云")
    args = parser.parse_args()

    # ---- 准备 ----
    set_chinese_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    font_path = get_wordcloud_font_path()
    print(f"📝 wordcloud 字体: {font_path}")

    stopwords = load_stopwords(STOPWORDS_PATH)
    print(f"📚 停用词加载: {len(stopwords)} 个")

    init_jieba(CUSTOM_DICT)
    print("🔧 jieba 自定义词典已加载")

    # ---- 数据 ----
    # 阶段1(词频/差异词):显式过滤机翻样本,避免"护照/班车/红色"等翻译伪影
    # 主导高频词。首次运行(sentiment_results.csv 还不存在)时自动跳过。
    limit = 100 if args.trial else None
    df = load_data(DATA_PATH, limit=limit, filter_translated=True)
    print(f"📥 数据加载: {len(df)} 条评论 {'【试跑模式】' if args.trial else ''}")

    df = tokenize_dataframe(df, stopwords)
    total_tokens = sum(len(ws) for ws in df["words"])
    print(f"✂️  分词完成: {total_tokens} 个有效词 token")

    c_all = count_words(df["words"])
    print(f"📊 整体词类: {len(c_all)} 个不重复词\n")

    # ---- 试跑分支:仅打印 Top20 + 一张词云 ----
    if args.trial:
        print("=== 试跑 Top20 ===")
        for i, (w, n) in enumerate(c_all.most_common(20), 1):
            print(f"  {i:>2}. {w:<10} {n}")
        wc_path = OUT_DIR / "all_wordcloud_trial.png"
        plot_wordcloud(c_all, str(wc_path), font_path, palette="panda")
        print(f"\n💾 试跑词云: {wc_path}")
        print("✅ 试跑完成。请检查 Top20 无垃圾词、词云中"
              "'大熊猫繁育研究基地'等专有词未被切碎,然后去掉 --trial 跑全量。")
        return

    # =====================================================
    # ===== 全量分析 =====
    # =====================================================
    # A. 整体
    print("[A] 整体分析")
    save_freq_csv(c_all, OUT_DIR / "all_words_freq.csv")
    print(f"  💾 {OUT_DIR / 'all_words_freq.csv'}")
    save_freq_csv(c_all, OUT_DIR / "all_top50.csv", top_n=50)
    print(f"  💾 {OUT_DIR / 'all_top50.csv'}")
    plot_bar(c_all, 50, "成都大熊猫基地游客评论高频词Top50(整体)",
             OUT_DIR / "all_top50_bar.png", NEUTRAL_COLOR)
    print(f"  🖼  {OUT_DIR / 'all_top50_bar.png'}")
    plot_wordcloud(c_all, str(OUT_DIR / "all_wordcloud.png"),
                   font_path, palette="panda")
    print(f"  ☁️  {OUT_DIR / 'all_wordcloud.png'}")

    # B. 分组
    print("\n[B] 分组分析")
    df_dom = df[df["group"] == "国内游客"]
    df_intl = df[df["group"] == "国际游客"]
    c_dom = count_words(df_dom["words"])
    c_intl = count_words(df_intl["words"])
    print(f"  国内 {len(df_dom)} 条 / {len(c_dom)} 词类;"
          f"国际 {len(df_intl)} 条 / {len(c_intl)} 词类")

    save_freq_csv(c_dom, OUT_DIR / "domestic_top50.csv", top_n=50)
    save_freq_csv(c_intl, OUT_DIR / "intl_top50.csv", top_n=50)
    print(f"  💾 domestic_top50.csv / intl_top50.csv")

    plot_bar(c_dom, 50, "国内游客高频词Top50",
             OUT_DIR / "domestic_top50_bar.png", DOM_COLOR)
    plot_bar(c_intl, 50, "国际游客高频词Top50",
             OUT_DIR / "intl_top50_bar.png", INTL_COLOR)
    print(f"  🖼  domestic_top50_bar.png / intl_top50_bar.png")

    plot_wordcloud(c_dom, str(OUT_DIR / "domestic_wordcloud.png"),
                   font_path, colormap="Reds")
    plot_wordcloud(c_intl, str(OUT_DIR / "intl_wordcloud.png"),
                   font_path, colormap="BuGn")
    print(f"  ☁️  domestic_wordcloud.png / intl_wordcloud.png")

    # C. 金字塔对比
    print("\n[C] 对比金字塔")
    plot_comparison(c_dom, c_intl, 30, OUT_DIR / "comparison_top30.png")
    print(f"  🖼  {OUT_DIR / 'comparison_top30.png'}")

    # D. 差异词
    print("\n[D] 差异词分析")
    df_dist_dom, df_dist_intl = find_distinctive_words(c_dom, c_intl,
                                                       top_n=200, smooth=5)
    df_dist_dom.to_csv(OUT_DIR / "distinctive_domestic.csv",
                       index=False, encoding="utf-8-sig")
    df_dist_intl.to_csv(OUT_DIR / "distinctive_intl.csv",
                        index=False, encoding="utf-8-sig")
    print(f"  💾 distinctive_domestic.csv / distinctive_intl.csv")
    plot_distinctive(df_dist_dom, df_dist_intl,
                     OUT_DIR / "distinctive_words.png")
    print(f"  🖼  {OUT_DIR / 'distinctive_words.png'}")

    # E. 词性分布
    print("\n[E] 词性分布")
    plot_pos_pie(df, 200, OUT_DIR / "pos_distribution.png")
    print(f"  🖼  {OUT_DIR / 'pos_distribution.png'}")

    # ---- 摘要打印 ----
    def _pretty(c, n=10):
        return [f"{w}({k})" for w, k in c.most_common(n)]

    print("\n" + "=" * 60)
    print("📌 整体 Top10:", "  ".join(_pretty(c_all)))
    print("📌 国内 Top10:", "  ".join(_pretty(c_dom)))
    print("📌 国际 Top10:", "  ".join(_pretty(c_intl)))
    print("\n📌 国内特征词 Top10:")
    for _, row in df_dist_dom.head(10).iterrows():
        print(f"   {row['word']:<8} 国内{row['国内频次']} / 国际{row['国际频次']} (比值 {row['比值']})")
    print("\n📌 国际特征词 Top10:")
    for _, row in df_dist_intl.head(10).iterrows():
        print(f"   {row['word']:<8} 国际{row['国际频次']} / 国内{row['国内频次']} (比值 {row['比值']})")

    # ---- 保存分词缓存(供阶段2 加速) ----
    # 阶段1 过滤了机翻样本,但阶段2 使用完整 2000 条。
    # 为使缓存对阶段2 可用,这里基于完整数据集重新分词并保存。
    from text_utils import save_tokens_cache
    df_full = load_data(DATA_PATH, filter_translated=False)
    if len(df_full) != len(df):
        df_full = tokenize_dataframe(df_full, stopwords)
        save_tokens_cache(df_full)
    else:
        save_tokens_cache(df)

    print(f"\n✅ 全部产出完成,共 14 个文件,在 {OUT_DIR}\\")


if __name__ == "__main__":
    main()
