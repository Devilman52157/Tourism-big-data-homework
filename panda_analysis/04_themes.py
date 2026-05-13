# -*- coding: utf-8 -*-
"""
04_themes.py
================
阶段4:AI 主题维度提炼 + 综合报告生成

能力
----
A. 汇总阶段1/2/3 的所有素材为结构化 Markdown(给 Gemini 的输入)
B. 调 Gemini 产出三段学术文字:
   B.1 目的地形象维度分析 (destination_image_analysis.md)
   B.2 国内 vs 国际 对比分析 (cross_cultural_comparison.md)
   B.3 研究结论与启示 (conclusions.md)
C. 基于 B.1 的维度划分做可视化:
   C.1 维度雷达图 (dimensions_radar.png)
   C.2 维度对照表 (dimensions_table.csv)
   C.3 主题词云分组图 (themes_grouped_wordcloud.png)
D. 合并为完整学术报告 (full_analysis_report.md)

用法
----
    python 04_themes.py --dump      # 只跑 A:汇总素材并把 prompt-input.md 打印出来
    python 04_themes.py --trial     # A + B.1:只跑维度分析,人工评估质量
    python 04_themes.py             # 全量:A + B.1 + B.2 + B.3 + C + D
    python 04_themes.py --offline   # 跳过 Gemini,仅基于已缓存的 md 重出可视化与合并报告
"""

import argparse
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import google.generativeai as genai  # noqa: E402

from font_config import set_chinese_font  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


# ============================================================
# 一、路径与常量
# ============================================================
DATA_PATH = "data/panda_sample_2000.csv"
OUT_DIR = Path("output/4_themes")

# 输入来源
IN_ALL_TOP50 = Path("output/1_word_freq/all_top50.csv")
IN_DOM_TOP50 = Path("output/1_word_freq/domestic_top50.csv")
IN_INT_TOP50 = Path("output/1_word_freq/intl_top50.csv")
IN_DIST_DOM = Path("output/1_word_freq/distinctive_domestic.csv")
IN_DIST_INT = Path("output/1_word_freq/distinctive_intl.csv")
IN_CENTRAL_ALL = Path("output/2_network/centrality_all.csv")
IN_SENTIMENT = Path("output/3_sentiment/sentiment_results.csv")
IN_TYPICAL = Path("output/3_sentiment/typical_reviews.csv")

# 输出
PROMPT_INPUT_MD = OUT_DIR / "_prompt_input.md"     # 给 Gemini 的素材
OUT_B1 = OUT_DIR / "destination_image_analysis.md"
OUT_B2 = OUT_DIR / "cross_cultural_comparison.md"
OUT_B3 = OUT_DIR / "conclusions.md"
OUT_RADAR = OUT_DIR / "dimensions_radar.png"
OUT_TABLE = OUT_DIR / "dimensions_table.csv"
OUT_WC = OUT_DIR / "themes_grouped_wordcloud.png"
OUT_FULL = OUT_DIR / "full_analysis_report.md"

MODEL_NAME = "gemini-2.5-flash"

# 8 个固定 aspect(与阶段3 保持一致)
ASPECTS_ORDER = ["熊猫互动", "环境景观", "服务设施", "票务交通",
                 "拥挤排队", "讲解科普", "餐饮购物", "性价比"]


# ============================================================
# 二、工具
# ============================================================
def _require(path: Path, hint: str):
    if not path.exists():
        print(f"[ERROR] 缺少依赖文件: {path}\n  提示:{hint}", file=sys.stderr)
        sys.exit(1)


def _comment_id_set(df: pd.DataFrame) -> set[int]:
    return set(pd.to_numeric(df["commentId"], errors="coerce")
               .dropna().astype("int64"))


def _validate_sentiment_matches_sample(sent_path: Path, sample_path: Path) -> None:
    """防止阶段4继续消费旧样本生成的 sentiment_results.csv。"""
    sample = pd.read_csv(sample_path, encoding="utf-8-sig", usecols=["commentId"])
    sent_ids_df = pd.read_csv(sent_path, encoding="utf-8-sig", usecols=["commentId"])
    sample_ids = _comment_id_set(sample)
    sent_ids = _comment_id_set(sent_ids_df)
    missing = sample_ids - sent_ids
    extra = sent_ids - sample_ids
    if missing or extra:
        print("[ERROR] sentiment_results.csv 与当前 panda_sample_2000.csv "
              "的 commentId 不一致。", file=sys.stderr)
        print(f"        当前样本 {len(sample_ids)} 条;情感结果 {len(sent_ids)} 条;"
              f"缺 {len(missing)} 条;多 {len(extra)} 条。", file=sys.stderr)
        print("        请先重跑 03_sentiment.py,再运行 04_themes.py。",
              file=sys.stderr)
        sys.exit(1)


def _split_aspects(cell) -> list:
    """aspects 在 sentiment_results.csv 里可能是 '|' 或 ',' 分隔,兼容两种。"""
    if pd.isna(cell) or not str(cell).strip():
        return []
    s = str(cell)
    parts = [x.strip() for x in re.split(r"[|,]", s) if x.strip()]
    return parts


def init_gemini():
    """初始化 Gemini 客户端(只在需要联网时才调)"""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("[ERROR] GEMINI_API_KEY 未配置", file=sys.stderr)
        sys.exit(1)
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        MODEL_NAME,
        generation_config={
            "temperature": 0.5,
            "max_output_tokens": 16384,
        },
    )


# ============================================================
# 三、A. 数据汇总 —— 给 Gemini 的 Markdown 输入
# ============================================================
def load_all_materials():
    """
    读取全部前置产出,返回一个 dict,便于后续既能写 markdown、又能做可视化。
    """
    _require(IN_ALL_TOP50, "请先跑 01_word_freq.py")
    _require(IN_DOM_TOP50, "请先跑 01_word_freq.py")
    _require(IN_INT_TOP50, "请先跑 01_word_freq.py")
    _require(IN_DIST_DOM, "请先跑 01_word_freq.py")
    _require(IN_DIST_INT, "请先跑 01_word_freq.py")
    _require(IN_CENTRAL_ALL, "请先跑 02_network.py")
    _require(IN_SENTIMENT, "请先跑 03_sentiment.py")
    _require(IN_TYPICAL, "请先跑 03_sentiment.py")
    _require(Path(DATA_PATH), "请先准备 data/panda_sample_2000.csv")
    _validate_sentiment_matches_sample(IN_SENTIMENT, Path(DATA_PATH))

    top50 = pd.read_csv(IN_ALL_TOP50, encoding="utf-8-sig")
    dom50 = pd.read_csv(IN_DOM_TOP50, encoding="utf-8-sig").head(30)
    int50 = pd.read_csv(IN_INT_TOP50, encoding="utf-8-sig").head(30)
    dist_dom = pd.read_csv(IN_DIST_DOM, encoding="utf-8-sig").head(20)
    dist_int = pd.read_csv(IN_DIST_INT, encoding="utf-8-sig").head(20)
    central = pd.read_csv(IN_CENTRAL_ALL, encoding="utf-8-sig").head(10)

    sent = pd.read_csv(IN_SENTIMENT, encoding="utf-8-sig")
    # 过滤非法行
    sent = sent[sent["sentiment"].isin(["正面", "负面", "中性"])].copy()
    sent["_aspects"] = sent["aspects"].apply(_split_aspects)

    typical = pd.read_csv(IN_TYPICAL, encoding="utf-8-sig")

    return {
        "top50": top50,
        "dom30": dom50,
        "int30": int50,
        "dist_dom": dist_dom,
        "dist_int": dist_int,
        "central": central,
        "sent": sent,
        "typical": typical,
    }


def compute_aspect_sentiment_stats(sent: pd.DataFrame) -> pd.DataFrame:
    """
    按 aspect × (整体/国内/国际) × sentiment 汇总计数,
    并给出情感得分 = (正-负)/总, 范围 [-1, 1]。
    返回长表:columns=[aspect, group, 正面, 中性, 负面, 总数, 情感得分]
    """
    rows = []
    for g_label, sub in [("整体", sent),
                         ("国内游客", sent[sent["group"] == "国内游客"]),
                         ("国际游客", sent[sent["group"] == "国际游客"])]:
        # 展平:每出现一个 aspect 就算一次
        expanded = []
        for _, r in sub.iterrows():
            for a in r["_aspects"]:
                if a in ASPECTS_ORDER:
                    expanded.append({"aspect": a, "sentiment": r["sentiment"]})
        if not expanded:
            for a in ASPECTS_ORDER:
                rows.append({"aspect": a, "group": g_label,
                             "正面": 0, "中性": 0, "负面": 0,
                             "总数": 0, "情感得分": 0.0})
            continue
        tmp = pd.DataFrame(expanded)
        tab = pd.crosstab(tmp["aspect"], tmp["sentiment"])
        for a in ASPECTS_ORDER:
            pos = int(tab.at[a, "正面"]) if a in tab.index and "正面" in tab.columns else 0
            neu = int(tab.at[a, "中性"]) if a in tab.index and "中性" in tab.columns else 0
            neg = int(tab.at[a, "负面"]) if a in tab.index and "负面" in tab.columns else 0
            total = pos + neu + neg
            score = (pos - neg) / total if total > 0 else 0.0
            rows.append({"aspect": a, "group": g_label,
                         "正面": pos, "中性": neu, "负面": neg,
                         "总数": total, "情感得分": round(score, 3)})
    return pd.DataFrame(rows)


def pick_reviews_for_prompt(typical: pd.DataFrame, k: int = 12) -> pd.DataFrame:
    """
    从 typical_reviews.csv 里均衡抽 k 条给 Gemini 当证据。
    策略:国内/国际 × 正/负 四象限各抽 3 条,尽量覆盖不同 aspect。
    """
    picks = []
    for g in ["国内游客", "国际游客"]:
        for s in ["正面", "负面"]:
            sub = typical[(typical["group"] == g) & (typical["sentiment"] == s)]
            if len(sub) == 0:
                continue
            # 按 aspect 做 drop_duplicates,尽量让抽样覆盖不同 aspect
            sub = sub.drop_duplicates(subset="aspect", keep="first")
            picks.append(sub.head(3))
    out = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()
    return out.head(k)


def build_prompt_input_md(m: dict) -> str:
    """构造给 Gemini 的结构化 markdown"""
    top50 = m["top50"]
    dom30, int30 = m["dom30"], m["int30"]
    dist_dom, dist_int = m["dist_dom"], m["dist_int"]
    central = m["central"]
    sent = m["sent"]

    # 预先算好情感统计
    aspect_stats = compute_aspect_sentiment_stats(sent)
    m["_aspect_stats"] = aspect_stats

    # 总体情感分布
    overall = sent["sentiment"].value_counts().reindex(
        ["正面", "中性", "负面"]).fillna(0).astype(int)
    n_total = int(overall.sum())
    n_dom = int((sent["group"] == "国内游客").sum())
    n_int = int((sent["group"] == "国际游客").sum())

    # 按 group 情感分布
    dom_sent = sent[sent["group"] == "国内游客"]["sentiment"].value_counts().reindex(
        ["正面", "中性", "负面"]).fillna(0).astype(int)
    int_sent = sent[sent["group"] == "国际游客"]["sentiment"].value_counts().reindex(
        ["正面", "中性", "负面"]).fillna(0).astype(int)

    lines = []
    lines.append("# 成都大熊猫繁育研究基地 游客评论分析素材汇总")
    lines.append("")
    lines.append(f"- 样本总量:**{n_total} 条**(国内游客 {n_dom} / 国际游客 {n_int})")
    lines.append(f"- 整体情感:正面 {overall['正面']} / 中性 {overall['中性']} / 负面 {overall['负面']}")
    lines.append("")

    # ---- 1. 整体 Top50 ----
    lines.append("## 1. 整体高频词 Top50")
    lines.append("")
    lines.append("| 排名 | 词 | 频次 | 占比 |   | 排名 | 词 | 频次 | 占比 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    rows = top50.head(50).reset_index(drop=True)
    for i in range(25):
        l = rows.iloc[i]
        r = rows.iloc[i + 25] if i + 25 < len(rows) else None
        left = f"| {l['rank']} | {l['word']} | {l['count']} | {l['ratio']} |"
        right = (f"| {r['rank']} | {r['word']} | {r['count']} | {r['ratio']} |"
                 if r is not None else "|  |  |  |  |")
        lines.append(left + "  " + right)
    lines.append("")

    # ---- 2. 国内 vs 国际 Top30 ----
    lines.append("## 2. 国内游客 Top30 vs 国际游客 Top30")
    lines.append("")
    lines.append("| 排名 | 国内词 | 国内频次 | 国际词 | 国际频次 |")
    lines.append("|---|---|---|---|---|")
    for i in range(30):
        dw = dom30.iloc[i] if i < len(dom30) else None
        iw = int30.iloc[i] if i < len(int30) else None
        lines.append(f"| {i+1} | {dw['word'] if dw is not None else ''} | "
                     f"{dw['count'] if dw is not None else ''} | "
                     f"{iw['word'] if iw is not None else ''} | "
                     f"{iw['count'] if iw is not None else ''} |")
    lines.append("")

    # ---- 3. 特征词 ----
    lines.append("## 3. 国内 / 国际 特征词 Top20(组间差异显著的词)")
    lines.append("")
    lines.append("### 3.1 国内游客特征词(国内高频、国际低频)")
    lines.append("")
    lines.append("| 排名 | 词 | 国内频次 | 国际频次 | 比值 |")
    lines.append("|---|---|---|---|---|")
    for i, (_, r) in enumerate(dist_dom.iterrows(), 1):
        lines.append(f"| {i} | {r['word']} | {r['国内频次']} | "
                     f"{r['国际频次']} | {r['比值']} |")
    lines.append("")
    lines.append("### 3.2 国际游客特征词(国际高频、国内低频)")
    lines.append("")
    lines.append("| 排名 | 词 | 国际频次 | 国内频次 | 比值 |")
    lines.append("|---|---|---|---|---|")
    for i, (_, r) in enumerate(dist_int.iterrows(), 1):
        lines.append(f"| {i} | {r['word']} | {r['国际频次']} | "
                     f"{r['国内频次']} | {r['比值']} |")
    lines.append("")

    # ---- 4. 中心度 ----
    lines.append("## 4. 整体社会语义网络 Top10 中心节点")
    lines.append("")
    lines.append("(说明:度中心性越高,表示该词与其它高频词共现范围越广,越处于网络核心)")
    lines.append("")
    lines.append("| 排名 | 词 | 度中心性 | 中介中心性 | 接近中心性 | 特征向量中心性 |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in central.iterrows():
        lines.append(f"| {int(r['rank_by_degree'])} | {r['word']} | "
                     f"{r['degree']:.3f} | {r['betweenness']:.3f} | "
                     f"{r['closeness']:.3f} | {r['eigenvector']:.3f} |")
    lines.append("")

    # ---- 5. 各 aspect 情感分布 ----
    lines.append("## 5. 8 个方面(aspect)的情感分布")
    lines.append("")
    lines.append("(情感得分 = (正面数 - 负面数) / 总数,范围 [-1, +1];数值越正表示该方面越被赞赏,越负表示越被吐槽)")
    lines.append("")
    lines.append("### 5.1 整体")
    lines.append("")
    lines.append("| 方面 | 正面 | 中性 | 负面 | 总提及 | 情感得分 |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in aspect_stats[aspect_stats["group"] == "整体"].iterrows():
        lines.append(f"| {r['aspect']} | {r['正面']} | {r['中性']} | "
                     f"{r['负面']} | {r['总数']} | {r['情感得分']} |")
    lines.append("")

    lines.append("### 5.2 国内 vs 国际 情感对比")
    lines.append("")
    lines.append("| 方面 | 国内得分 | 国内(正/中/负/总) | 国际得分 | 国际(正/中/负/总) | 差值(国内-国际) |")
    lines.append("|---|---|---|---|---|---|")
    dom_rows = aspect_stats[aspect_stats["group"] == "国内游客"].set_index("aspect")
    int_rows = aspect_stats[aspect_stats["group"] == "国际游客"].set_index("aspect")
    for a in ASPECTS_ORDER:
        d = dom_rows.loc[a]
        i = int_rows.loc[a]
        diff = round(float(d["情感得分"]) - float(i["情感得分"]), 3)
        lines.append(
            f"| {a} | {d['情感得分']} | "
            f"{int(d['正面'])}/{int(d['中性'])}/{int(d['负面'])}/{int(d['总数'])} | "
            f"{i['情感得分']} | "
            f"{int(i['正面'])}/{int(i['中性'])}/{int(i['负面'])}/{int(i['总数'])} | "
            f"{diff:+.3f} |"
        )
    lines.append("")

    # ---- 6. 国内 / 国际 情感分布总览 ----
    lines.append("## 6. 国内 vs 国际 整体情感分布")
    lines.append("")
    lines.append(f"- 国内游客({n_dom} 条):正面 {dom_sent['正面']} / 中性 {dom_sent['中性']} / 负面 {dom_sent['负面']}")
    lines.append(f"- 国际游客({n_int} 条):正面 {int_sent['正面']} / 中性 {int_sent['中性']} / 负面 {int_sent['负面']}")
    lines.append("")

    # ---- 7. 典型评论 ----
    picks = pick_reviews_for_prompt(m["typical"], k=12)
    lines.append("## 7. 代表性评论抽样(12 条,4 象限均衡)")
    lines.append("")
    for _, r in picks.iterrows():
        txt = str(r["content_clean"]).replace("\n", " ")
        if len(txt) > 240:
            txt = txt[:240] + "..."
        lines.append(f"- **[{r['group']} · {r['sentiment']} · {r['aspect']}]** "
                     f"(key=「{r['key_phrase']}」)")
        lines.append(f"  > {txt}")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# 四、B. 三段学术文字 prompt
# ============================================================
B1_PROMPT = """你是一名旅游学领域的专家研究员,擅长目的地形象(Destination Image)研究和旅游文本挖掘。

现在你拿到一份成都大熊猫繁育研究基地的游客评论分析数据(共 2000 条评论,国内国际各 1000),请基于以下材料,撰写一段约 800 字的"目的地形象感知维度分析"。

【材料】
{material}

【要求】
1. 将材料 1 中的 50 个高频词归类成 **正好 5 个主题维度**,每个维度起一个学术化的命名(如"核心吸引物""景观环境""服务运营""教育科普""感知评价"等,具体名称你根据数据判断)。
2. 每个维度控制在 130-160 字,结构固定如下:
   - `### N. 维度名`(h3 标题,N 为 1-5 的序号)
   - `**包含高频词**`:列出 5-8 个词及其频次,格式 `词(频次)`,顿号分隔
   - `**维度解读**`:2-3 句话,说明该维度反映了游客什么样的感知与诉求
   - `**情感倾向**`:结合材料 5 的 aspect 情感得分,指出该维度整体是被赞赏还是被吐槽,给量化依据;若国内国际差异显著,要点明
   - `**典型证据**`:引用材料 7 中 1 条真实评论片段即可,不要超过 80 字
3. 5 个维度写完后,以 `## 总体形象小结` 为二级标题,再写 130-180 字的总结,回答:**游客对成都大熊猫基地的核心目的地形象是什么?** 应当落到 2-3 个形象标签上(如"可爱的国宝+大而拥挤的景区+中等性价比"),结合数据给出结论。这部分是必须完整输出的,不要省略。
4. 写作风格:参考《旅游学刊》《旅游科学》的学术风格,用词学术化但不晦涩;避免"综上所述""值得注意的是""总而言之"等套话;每个论点都必须用材料中的具体数字或评论作为证据。

【输出格式】
- 顶部一级标题 `# 一、目的地形象感知维度分析`
- 5 个维度用 h3(`###`)子标题,带序号
- 结尾"总体形象小结"用 h2(`##`)子标题
- 直接输出 Markdown 正文,不要加代码块围栏,不要输出说明性的前后缀
- **严格控制总字数在 900 字以内**,确保最后的总结段完整
"""

B2_PROMPT = """你是一名旅游学领域的专家研究员,擅长跨文化旅游行为研究。

基于下列材料中的【国内/国际特征词】(材料 3)、【国内/国际整体情感分布】(材料 6)、以及【8 个方面的国内 vs 国际情感差异】(材料 5.2),撰写一段约 600 字的"国内外游客感知差异分析"。

【材料】
{material}

【要求】
1. 用材料 3 的特征词对比,指出**关注点的差异**(国内在意什么、国际在意什么)。必须点名具体特征词并引用其比值。
2. 用材料 5.2 的情感得分**差值**,找出 2–3 个"两组感知最分歧的方面",并用数据说明谁更正面谁更负面。
3. 用材料 6 的整体正负面比例,说明**两组总体评价极性**的差异。
4. 从"文化预期""信息渠道""消费习惯""语言与翻译"中选 2 条角度,对差异做深层解释;解释要贴合数据,不要空泛。
5. 最后给景区管理方 1–2 条**可操作建议**,必须针对国内/国际两组中的**薄弱环节**。
6. 不要使用"值得注意的是""综上所述""总而言之"等套话;每个论点必须用材料中的数字或评论作为支撑。

【输出格式】
- 一级标题 `# 二、国内外游客感知差异分析`
- 小节可自由组织,建议按"关注点差异 → 情感极性差异 → 深层解释 → 管理建议"展开
- 直接输出 Markdown 正文,不要加代码块围栏
"""

B3_PROMPT = """你是一名旅游学领域的专家研究员。

基于你刚刚写的两段分析(B.1 目的地形象维度 + B.2 国内外差异),以及下列原始材料,撰写一段约 400 字的"研究结论与启示"。

【B.1 维度分析结果】
{b1}

【B.2 差异分析结果】
{b2}

【核心原始数据】
{material}

【要求】
1. **核心发现(3–4 点)**:每条发现必须落到具体数字或维度上,避免泛泛而谈。
2. **理论贡献(1 点)**:对目的地形象理论或 ABSA 方法,本研究能提供什么新观察。
3. **管理启示(2–3 条)**:每条对应一个可操作的改进方向,不要泛泛说"加强服务"。
4. **研究局限与未来方向(1–2 点)**:点出本研究在数据源、方法、样本上的明确局限。
5. 不要套话,不要"综上所述""展望未来"等开场。
6. 总长度控制在 400 字左右。

【输出格式】
- 一级标题 `# 三、研究结论与启示`
- 下分 4 个 h2 小节:核心发现 / 理论贡献 / 管理启示 / 研究局限
- 直接输出 Markdown 正文,不要加代码块围栏
"""


def call_gemini(model, prompt: str, label: str) -> str:
    """一次 Gemini 调用,做简单的失败兜底。"""
    print(f"\n[AI] 调用 Gemini 生成 {label} ...")
    try:
        resp = model.generate_content(prompt)
        text = resp.text if hasattr(resp, "text") else str(resp)
        if not text or not text.strip():
            raise ValueError("Gemini 返回空内容")
        # 有时模型会多包一层 ```markdown\n...\n```,剥掉
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()
        print(f"[AI] {label} 生成完毕,{len(text)} 字")
        return text
    except Exception as e:
        print(f"[ERROR] {label} 生成失败:{e}", file=sys.stderr)
        raise


# ============================================================
# 五、C. 可视化产出
# ============================================================
def parse_dimensions_from_b1(b1_text: str) -> list:
    """
    从 B.1 markdown 里提取维度划分:
    返回 [{name: "xxx", words: [(词, 频次), ...]}, ...]
    约定 B.1 的格式:
        ### 维度名
        - **包含高频词**:词1(n1)、词2(n2)、...
      兼容中文冒号和中文括号:
        **包含高频词**：词1（n1）、词2（n2）、...
    """
    dims = []
    # 按 ### 分段
    chunks = re.split(r"^###\s+", b1_text, flags=re.MULTILINE)
    for c in chunks[1:]:
        # 第一行是维度名(可能带"维度""(x词)"等后缀)
        name_line, _, rest = c.partition("\n")
        name = name_line.strip().strip(":").strip()
        # 找到"包含高频词"这一行
        m = re.search(
            r"(?:[-*]\s*)?\*{0,2}\s*包含高频词\s*\*{0,2}\s*[:：]\s*(.+)",
            rest,
        )
        if not m:
            continue
        word_line = m.group(1)
        # 解析 "熊猫(1119)、大熊猫(179)、..." 允许全/半角括号和中文顿号/逗号
        items = re.findall(
            r"([\u4e00-\u9fffA-Za-z0-9_]+)\s*[\(（]\s*(\d+)\s*[\)）]",
            word_line,
        )
        words = [(w, int(n)) for w, n in items]
        if words:
            dims.append({"name": name, "words": words})
    return dims


def build_dimensions_table(dims: list, m: dict) -> pd.DataFrame:
    """
    对每个维度,聚合它所覆盖 aspect 上的情感得分:
      - 思路:维度里包含的高频词,映射回"这些词所在评论"的集合,
        再在该集合上按 group 算情感得分
      - 若维度里没词命中评论(罕见),置 0
    """
    sent = m["sent"]
    # 预索引:每条评论的词集(简单用 content_clean 的 substring 判断)
    # 精度: Top50 词都是较长的实义词,直接 in 判断即可满足
    rows = []
    for dim in dims:
        words = [w for w, _ in dim["words"]]
        if not words:
            continue
        mask = pd.Series([False] * len(sent), index=sent.index)
        content = sent["content_clean"].astype(str)
        for w in words:
            mask = mask | content.str.contains(re.escape(w), na=False)
        sub = sent[mask]

        def _score(df_):
            pos = (df_["sentiment"] == "正面").sum()
            neg = (df_["sentiment"] == "负面").sum()
            total = len(df_)
            return round((pos - neg) / total, 3) if total > 0 else 0.0

        all_score = _score(sub)
        dom_score = _score(sub[sub["group"] == "国内游客"])
        int_score = _score(sub[sub["group"] == "国际游客"])

        typical_quote = ""
        if len(sub) > 0:
            # 统计每条评论命中本维度词的个数,优先选命中多的短评论
            # (长度太长的评论通常是啥都讲的综合评论,不够"聚焦"某一维度)
            def _match_count(text):
                t = str(text)
                return sum(1 for w in words if w in t)

            cand = sub.copy()
            cand["_hits"] = cand["content_clean"].map(_match_count)
            cand["_len"] = cand["content_clean"].astype(str).map(len)
            # 先过滤太长的(避免一条长评论被每个维度都命中)
            short = cand[cand["_len"] <= 200]
            pool = short if len(short) > 0 else cand
            # 按命中数降序、长度升序
            pool = pool.sort_values(["_hits", "_len"], ascending=[False, True]).head(1)
            if len(pool):
                txt = str(pool.iloc[0]["content_clean"]).replace("\n", " ")
                if len(txt) > 120:
                    txt = txt[:120] + "..."
                typical_quote = txt

        rows.append({
            "维度名称": dim["name"],
            "包含的高频词": "、".join(f"{w}({n})" for w, n in dim["words"]),
            "覆盖评论数": int(len(sub)),
            "整体情感得分": all_score,
            "国内情感得分": dom_score,
            "国际情感得分": int_score,
            "典型评论": typical_quote,
        })
    return pd.DataFrame(rows)


def plot_dimensions_radar(df_table: pd.DataFrame, save_path: Path):
    """维度雷达图:国内 vs 国际 情感得分"""
    if df_table.empty:
        print("  [WARN] 维度表为空,跳过雷达图")
        return
    labels = df_table["维度名称"].tolist()
    dom = df_table["国内情感得分"].astype(float).tolist()
    intl = df_table["国际情感得分"].astype(float).tolist()

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    # 闭环
    dom_c = dom + [dom[0]]
    intl_c = intl + [intl[0]]
    angles_c = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150,
                           subplot_kw=dict(polar=True))
    ax.plot(angles_c, dom_c, "o-", linewidth=2.2, color="#E74C3C", label="国内游客")
    ax.fill(angles_c, dom_c, color="#E74C3C", alpha=0.18)
    ax.plot(angles_c, intl_c, "s-", linewidth=2.2, color="#1ABC9C", label="国际游客")
    ax.fill(angles_c, intl_c, color="#1ABC9C", alpha=0.18)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=12)
    # 情感得分范围 [-1, 1]
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(["-1", "-0.5", "0", "0.5", "+1"], fontsize=9)
    ax.set_rlabel_position(180 / len(labels))
    ax.grid(True, linestyle="--", alpha=0.5)

    # 标注每个点数值
    for ang, v in zip(angles, dom):
        ax.text(ang, v + 0.05, f"{v:+.2f}", ha="center",
                fontsize=9, color="#C0392B")
    for ang, v in zip(angles, intl):
        ax.text(ang, v - 0.08, f"{v:+.2f}", ha="center",
                fontsize=9, color="#117864")

    plt.title("维度层面情感得分对比(国内 vs 国际)", fontsize=15, pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.08), fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_themes_grouped_wordcloud(dims: list, save_path: Path):
    """
    把每个维度的词做一个小词云,拼成一张大图(2 列布局)。
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("  [WARN] wordcloud 未安装,跳过分组词云")
        return

    # 中文字体路径:统一从 font_config 获取
    from font_config import get_wordcloud_font_path
    try:
        font_path = get_wordcloud_font_path()
    except FileNotFoundError:
        print("  [WARN] 未找到中文字体文件,跳过分组词云")
        return

    if not dims:
        print("  [WARN] 维度为空,跳过分组词云")
        return

    n = len(dims)
    cols = 2 if n >= 2 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.5, rows * 4.5),
                             dpi=150)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    # 每个维度一个配色
    palettes = ["Reds", "BuGn", "Blues", "Oranges", "Purples", "PuRd"]

    for idx, dim in enumerate(dims):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]
        freqs = dict(dim["words"])
        if not freqs:
            ax.axis("off")
            continue
        wc = WordCloud(
            font_path=font_path,
            width=800, height=500,
            background_color="white",
            colormap=palettes[idx % len(palettes)],
            max_words=len(freqs), min_font_size=18, max_font_size=100,
            prefer_horizontal=0.95, random_state=42, collocations=False,
            margin=6,
        ).generate_from_frequencies(freqs)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(dim["name"], fontsize=14, pad=8)
        ax.axis("off")

    # 隐藏多余子图
    for idx in range(n, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis("off")

    fig.suptitle("目的地形象维度 — 主题词分组词云", fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 六、D. 综合报告
# ============================================================
REPORT_FRONT_MATTER = """# 成都大熊猫繁育研究基地游客评论分析报告

## 摘要

本研究以 2000 条成都大熊猫繁育研究基地游客评论为样本(国内游客与国际游客各 1000 条),综合运用词频统计、社会语义网络分析与基于大语言模型(Gemini 2.5 Flash)的方面级情感分析(ABSA),对该景区的目的地形象、情感评价与跨文化差异进行系统研究。在此基础上,本文从 5 个主题维度提炼景区的形象感知结构,并对比国内外游客的关注点与情感极性差异,为景区运营管理提供量化依据。

**关键词**:成都大熊猫基地;目的地形象;方面级情感分析;社会语义网络;跨文化比较;大语言模型

## 研究方法说明

- **数据来源**:携程(Trip.com)景区评论,经字段清洗后获得 2000 条有效样本,按 IP 归属与平台字段划分"国内游客"与"国际游客"各 1000 条。
- **分词与词频分析**:基于 jieba 分词并结合自定义词典与停用词表,保留名词、动词、形容词三类实义词,输出整体及分组 Top50 高频词与差异词。
- **社会语义网络分析**:以整体 Top30 词为节点,以"同评论共现"为边权,计算度中心性、中介中心性、接近中心性与特征向量中心性四种指标,并使用 Louvain 算法检测语义社区。
- **方面级情感分析**:设定 8 个固定方面(熊猫互动、环境景观、服务设施、票务交通、拥挤排队、讲解科普、餐饮购物、性价比),调用 Gemini 2.5 Flash 对每条评论进行情感极性(正面/负面/中性)、情感强度、涉及方面与机翻识别等多字段标注,批量成功率 100%。
- **主题维度提炼**:由 Gemini 扮演旅游学研究专家,基于前述全部数据对 Top50 高频词进行 5 个主题维度的归类,并撰写目的地形象分析、跨文化差异分析与研究结论。

---
"""


def build_full_report(b1: str, b2: str, b3: str, save_path: Path):
    parts = [REPORT_FRONT_MATTER, b1.strip(), "\n\n---\n",
             b2.strip(), "\n\n---\n", b3.strip(), ""]
    save_path.write_text("\n\n".join(parts), encoding="utf-8")
    print(f"  📝 {save_path}")


# ============================================================
# 七、主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="阶段4:AI 主题维度提炼与综合报告")
    parser.add_argument("--dump", action="store_true",
                        help="只跑 A:汇总素材并写出 _prompt_input.md,供人工审查")
    parser.add_argument("--trial", action="store_true",
                        help="A + B.1 试跑:只生成维度分析,评估质量")
    parser.add_argument("--offline", action="store_true",
                        help="跳过 Gemini,基于已有 b1/b2/b3 md 只重做可视化与综合报告")
    args = parser.parse_args()

    set_chinese_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[A] 汇总前置阶段素材")
    materials = load_all_materials()
    prompt_md = build_prompt_input_md(materials)
    PROMPT_INPUT_MD.write_text(prompt_md, encoding="utf-8")
    print(f"  💾 {PROMPT_INPUT_MD}  ({len(prompt_md):,} 字)")

    if args.dump:
        print("\n=== --dump 模式:仅汇总素材,打印并退出 ===\n")
        # 只打印前 120 行到控制台,避免刷屏
        preview = "\n".join(prompt_md.splitlines()[:120])
        print(preview)
        print("\n...(完整内容见 _prompt_input.md)...")
        return

    if args.offline:
        print("\n[offline] 跳过 Gemini,基于已有 b1/b2/b3 md 重做可视化与报告")
        for p in [OUT_B1, OUT_B2, OUT_B3]:
            _require(p, f"请先跑一次完整的 04_themes.py 生成 {p.name}")
        b1 = OUT_B1.read_text(encoding="utf-8")
        b2 = OUT_B2.read_text(encoding="utf-8")
        b3 = OUT_B3.read_text(encoding="utf-8")
    else:
        model = init_gemini()

        # ---- B.1 ----
        b1 = call_gemini(model, B1_PROMPT.format(material=prompt_md), "B.1 维度分析")
        OUT_B1.write_text(b1, encoding="utf-8")
        print(f"  📝 {OUT_B1}")

        if args.trial:
            print("\n=== --trial 模式:只跑 B.1,人工检查 destination_image_analysis.md ===")
            print("满意后去掉 --trial 再跑,即可继续 B.2 / B.3 / 可视化 / 综合报告")
            return

        # ---- B.2 ----
        b2 = call_gemini(model, B2_PROMPT.format(material=prompt_md), "B.2 跨文化对比")
        OUT_B2.write_text(b2, encoding="utf-8")
        print(f"  📝 {OUT_B2}")

        # ---- B.3 ----
        b3 = call_gemini(model,
                         B3_PROMPT.format(material=prompt_md, b1=b1, b2=b2),
                         "B.3 研究结论")
        OUT_B3.write_text(b3, encoding="utf-8")
        print(f"  📝 {OUT_B3}")

    # ---- C. 可视化 ----
    print("\n[C] 可视化")
    dims = parse_dimensions_from_b1(b1)
    print(f"  从 B.1 解析到 {len(dims)} 个维度:"
          f"{', '.join(d['name'] for d in dims)}")

    if not dims:
        print("  [WARN] 未能从 B.1 解析出维度,可视化将被跳过")
    else:
        df_table = build_dimensions_table(dims, materials)
        df_table.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")
        print(f"  💾 {OUT_TABLE}")

        plot_dimensions_radar(df_table, OUT_RADAR)
        print(f"  🖼 {OUT_RADAR}")

        plot_themes_grouped_wordcloud(dims, OUT_WC)
        print(f"  🖼 {OUT_WC}")

    # ---- D. 合并综合报告 ----
    print("\n[D] 生成综合报告")
    build_full_report(b1, b2, b3, OUT_FULL)

    # ---- 清单 ----
    official = [OUT_B1, OUT_B2, OUT_B3, OUT_RADAR, OUT_TABLE, OUT_WC, OUT_FULL]
    print("\n" + "=" * 60)
    print(f"📦 产出清单  ({OUT_DIR}/)")
    for p in official:
        if p.exists():
            size_kb = p.stat().st_size / 1024
            print(f"  ✓ {p.name:<42}  {size_kb:>8.1f} KB")
        else:
            print(f"  ✗ {p.name:<42}  (缺失)")


if __name__ == "__main__":
    main()
