# -*- coding: utf-8 -*-
"""
02_network.py
================
阶段2:社会语义网络分析(整体 / 国内 / 国际 三组对比)

核心逻辑:
    1) 复用阶段1的分词流程(load_stopwords / init_jieba / tokenize_dataframe)
    2) 以整体 Top30 词作为三组网络的统一节点集
    3) 共现窗口 = 整条评论(一条评论里两词共同出现 → 共现次数 +1;
       同一条评论里同一对词去重,不受重复次数影响)
    4) 对三组网络分别算度/中介/接近/特征向量中心性
    5) matplotlib 画静态图、pyvis 画交互图、国内vs国际左右对比图、
       Louvain 社区分析 Markdown 报告

用法:
    python 02_network.py            # 全量(整体/国内/国际三组完整产出)
    python 02_network.py --trial    # 试跑:只产出 network_static_all.png
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community import louvain_communities

# pyvis 交互网络
from pyvis.network import Network

# 共用工具 —— 从阶段间共用模块导入,不再用 importlib 动态加载 01
from text_utils import (
    load_stopwords, init_jieba, load_data,
    tokenize_dataframe, count_words,
)
from font_config import set_chinese_font

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


# ============================================================
# 一、配置常量
# ============================================================
DATA_PATH = "data/panda_sample_2000.csv"
CUSTOM_DICT = "custom_dict.txt"
STOPWORDS_PATH = "stopwords.txt"
OUT_DIR = Path("output/2_network")

# 核心可调参数(有推荐区间,不要越界)
TOP_N = 30              # 节点数,推荐 20~50

# 边过滤阈值——分两层处理:
# (1) 整体网络:保留经验值 25。2000 条评论里 Top30 全是高频词,
#     两两共现的中位数就达到 16,阈值 3 会画成几乎全连接的糊图。
#     在 25 附近保留约 140 条边,Louvain 稳定分出 3 个社区。
# (2) 分组网络:用各自共现矩阵上三角非零值的 EDGE_PERCENTILE 分位数
#     作为阈值。原先固定 25 用在分组网络上会导致国内/国际分别产生
#     8/7 个社区(且半数是孤立单节点)——分组样本量本就小一半,
#     再经机翻过滤后国际组只剩约 35%,固定阈值已不再合适。
EDGE_THRESHOLD = 25
EDGE_PERCENTILE = 70    # 分组网络的动态阈值分位数,实测 65~75 都能落到 3-5 个社区

# Louvain 随机种子与布局种子 —— 保证跨组网络、多次运行一致
LOUVAIN_SEED = 42
LAYOUT_SEED = 42

# 社区调色板(最多 10 个颜色,够用;社区数超过 10 说明数据稀疏,需提示用户)
COMMUNITY_PALETTE = [
    "#E74C3C", "#1ABC9C", "#F1C40F", "#3498DB", "#9B59B6",
    "#2ECC71", "#E67E22", "#34495E", "#E84393", "#16A085",
]

# 视觉编码参数
# 节点大小用 sqrt 映射,比 log 保留更多差异(熊猫 2170 vs 推荐 116 → sqrt 差 4 倍,log 只差 1.5 倍)
NODE_BASE_SIZE = 120       # 节点最小面积
NODE_SCALE = 42            # 节点面积随 sqrt(freq) 放大系数
EDGE_BASE_WIDTH = 0.4      # 边最小宽度
EDGE_SCALE = 0.05          # 边宽度随共现次数放大系数(阈值=25,再大会太粗)

# spring_layout 参数 —— k 越大节点越分散,iterations 越高越稳定
LAYOUT_K = 1.2
LAYOUT_ITERS = 200


# ============================================================
# 二、共现矩阵
# ============================================================
def build_cooccurrence_matrix(tokenized_reviews, node_words):
    """
    基于"整条评论"窗口构建共现矩阵。

    逐步骤解释(报告里要讲清楚的就是这里):

        【第1步:节点集来源】
            node_words 由调用方传入 —— 本项目里它固定为"整体 2000 条评论
            的 Top30 高频词",三组网络共用,这样对比图才公平。

        【第2步:共现窗口定义】
            窗口 = 整条评论。也就是说,只要某条评论的分词结果里同时出现
            node_words 中的两个词,这一对词就被视为"共现一次"。
            这不同于"滑动窗口 N 个词内共现"的做法,更符合"游客在一句
            描述里整体联想的概念"。

        【第3步:同评论同对词去重】
            同一条评论里"熊猫"出现 3 次、"可爱"出现 2 次,
            (熊猫, 可爱) 的共现次数只 +1,而不是 +6。
            这避免了长评论对共现强度的过度贡献。

        【第4步:矩阵对称填充】
            共现是无向的,M[a][b] 与 M[b][a] 同时 +1,对角线永远 0。

    参数
    ----
    tokenized_reviews : Iterable[List[str]]
        已分词的评论列表,每个元素是一条评论的词列表(阶段1的 words 列)。
    node_words : List[str]
        节点集。矩阵行列标签与顺序完全跟随该列表。

    返回
    ----
    pd.DataFrame
        对称的共现次数矩阵,index / columns 均为 node_words。
    """
    n = len(node_words)
    idx = {w: i for i, w in enumerate(node_words)}
    node_set = set(node_words)

    # 用 int64 矩阵,2000 条评论远没到 int32 溢出风险
    M = np.zeros((n, n), dtype=np.int64)

    from itertools import combinations

    for words in tokenized_reviews:
        # 第3步:同评论同对词去重 → set()
        present = set(words) & node_set
        if len(present) < 2:
            continue
        # 所有两两组合 +1(矩阵上下三角同时更新,保证对称)
        for wa, wb in combinations(present, 2):
            i, j = idx[wa], idx[wb]
            M[i, j] += 1
            M[j, i] += 1

    return pd.DataFrame(M, index=node_words, columns=node_words)


def dynamic_edge_threshold(cooc_df, percentile=EDGE_PERCENTILE, min_thr=3):
    """从共现矩阵的上三角非零值取分位数作为分组网络的边阈值。

    返回 (threshold, n_nonzero_edges):
        - threshold:四舍五入后的整数阈值,不低于 min_thr
        - n_nonzero_edges:上三角非零边数(用于日志)
    """
    M = cooc_df.values
    n = M.shape[0]
    vals = M[np.triu_indices(n, k=1)]
    vals = vals[vals > 0]
    if len(vals) == 0:
        return min_thr, 0
    thr = int(round(np.percentile(vals, percentile)))
    return max(thr, min_thr), len(vals)


def build_graph_from_matrix(cooc_df, edge_threshold):
    """
    把共现矩阵转成 networkx.Graph:
        - 每个节点词都加进图(即使无边,也要出现在节点集里,保证三组网络节点一致)
        - 仅当共现次数 >= edge_threshold 时加边,边权 = 共现次数
    """
    G = nx.Graph()
    for w in cooc_df.index:
        G.add_node(w)
    cols = list(cooc_df.columns)
    M = cooc_df.values
    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
            w_ij = int(M[i, j])
            if w_ij >= edge_threshold:
                G.add_edge(cols[i], cols[j], weight=w_ij)
    return G


# ============================================================
# 四、中心度
# ============================================================
def compute_centrality(G, group_label=""):
    """
    对一个图计算 4 种中心度。
    权重处理:
        - degree_centrality:不加权(反映"连接词数")
        - betweenness / closeness:以 1/weight 作为"距离",
          这样共现越强的边在最短路径里"更短",符合"桥梁作用"的语义
        - eigenvector:直接用共现次数作为权,"强连接 → 更重要"

    返回
    ----
    pd.DataFrame,列:word, degree, betweenness, closeness, eigenvector, rank_by_degree
    """
    nodes = list(G.nodes())

    # 构造一个带 distance 属性的副本,专门给 betweenness / closeness 用
    G_dist = G.copy()
    for u, v, d in G_dist.edges(data=True):
        w = d.get("weight", 1)
        d["distance"] = 1.0 / w if w > 0 else 1.0

    deg = nx.degree_centrality(G)
    btw = nx.betweenness_centrality(G_dist, weight="distance")
    # closeness 支持 distance= 参数
    cls = nx.closeness_centrality(G_dist, distance="distance")

    # eigenvector:稀疏或不连通时可能不收敛,兜底为全 0
    # 用纯 Python 幂迭代版本,不依赖 scipy
    try:
        evc = nx.eigenvector_centrality(G, weight="weight", max_iter=1000, tol=1e-6)
    except Exception as e:
        print(f"[WARN] ({group_label}) eigenvector_centrality 计算失败"
              f"({e.__class__.__name__}: {e}),该列填 0")
        evc = {n: 0.0 for n in nodes}

    df = pd.DataFrame({
        "word": nodes,
        "degree": [deg.get(n, 0.0) for n in nodes],
        "betweenness": [btw.get(n, 0.0) for n in nodes],
        "closeness": [cls.get(n, 0.0) for n in nodes],
        "eigenvector": [evc.get(n, 0.0) for n in nodes],
    })
    df = df.sort_values("degree", ascending=False).reset_index(drop=True)
    df["rank_by_degree"] = np.arange(1, len(df) + 1)
    return df


# ============================================================
# 五、社区检测
# ============================================================
def detect_communities(G, seed=LOUVAIN_SEED):
    """
    Louvain 社区检测。
    返回 (word -> community_id) 的 dict;孤立节点(度为 0)归为独立社区。
    """
    # 孤立节点从 Louvain 中单拎出来,避免污染社区划分
    isolates = [n for n, d in G.degree() if d == 0]
    G_core = G.subgraph([n for n in G.nodes() if n not in isolates]).copy()

    if G_core.number_of_nodes() == 0:
        # 极端情况:所有节点都孤立
        return {n: i for i, n in enumerate(G.nodes())}

    comms = louvain_communities(G_core, weight="weight", seed=seed)
    word2cid = {}
    for cid, members in enumerate(comms):
        for w in members:
            word2cid[w] = cid

    # 孤立节点追加为新社区,每个一档
    next_cid = len(comms)
    for w in isolates:
        word2cid[w] = next_cid
        next_cid += 1
    return word2cid


# ============================================================
# 六、静态网络图(matplotlib)
# ============================================================
def _node_size_by_freq(nodes, overall_freq):
    """节点面积 = base + scale * sqrt(freq),用整体词频做映射。
    相比 log,sqrt 保留更多高低频差异,同时不会让最高频词爆炸。"""
    return [NODE_BASE_SIZE + NODE_SCALE * np.sqrt(overall_freq.get(n, 0))
            for n in nodes]


def _edge_widths(G):
    widths = []
    for u, v, d in G.edges(data=True):
        widths.append(EDGE_BASE_WIDTH + EDGE_SCALE * d.get("weight", 1))
    return widths


def _community_colors(nodes, word2cid):
    """给每个节点取颜色,社区 id 循环映射到调色板。"""
    cids = sorted({word2cid[n] for n in nodes})
    cid2color = {cid: COMMUNITY_PALETTE[i % len(COMMUNITY_PALETTE)]
                 for i, cid in enumerate(cids)}
    return [cid2color[word2cid[n]] for n in nodes], cid2color


def plot_static_network(G, overall_freq, word2cid, pos, title, save_path,
                        edge_threshold=EDGE_THRESHOLD):
    """
    为一个图画静态 PNG。
    参数
    ----
    G : nx.Graph            —— 已按阈值过滤后的图
    overall_freq : dict     —— 词 -> 整体词频(用来映射节点大小)
    word2cid : dict         —— 词 -> 社区 id(用来上色)
    pos : dict              —— 节点 -> (x,y),三组共用(确保对比公平)
    title : str             —— 标题
    save_path : Path        —— 保存路径
    """
    nodes = list(G.nodes())
    sizes = _node_size_by_freq(nodes, overall_freq)
    colors, cid2color = _community_colors(nodes, word2cid)
    widths = _edge_widths(G)

    # 1600x1200 像素 @ dpi=150 → figsize ≈ 10.67 x 8
    fig, ax = plt.subplots(figsize=(10.67, 8), dpi=150)

    # 先画边(浅灰),再画节点覆在上面
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=widths, edge_color="#9e9e9e", alpha=0.55,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, nodelist=nodes,
        node_size=sizes, node_color=colors,
        edgecolors="white", linewidths=1.2, alpha=0.95,
    )
    # 中文标签 —— 走全局 rcParams['font.sans-serif'] 即可
    # 给标签加白色半透明底:小节点 + 多字词标签会"溢出"圆圈,这样仍清晰可读
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=10, font_color="black",
        bbox=dict(boxstyle="round,pad=0.18",
                  facecolor="white", edgecolor="none", alpha=0.78),
    )

    ax.set_title(title, fontsize=15, pad=14)
    ax.axis("off")

    # 图例:节点大小 / 边粗细 / 颜色说明
    legend_text = (
        "节点大小:整体词频(sqrt 缩放)\n"
        "边粗细:共现次数(≥ {thr} 才画)\n"
        "节点颜色:Louvain 社区(共 {ncom} 个)"
    ).format(thr=edge_threshold, ncom=len(cid2color))
    ax.text(
        0.02, 0.02, legend_text,
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="white", edgecolor="#cccccc", alpha=0.85),
        verticalalignment="bottom",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 七、交互式网络图(pyvis)
# ============================================================
def plot_interactive_network(G, overall_freq, word2cid, centrality_df,
                             title, save_path):
    """
    用 pyvis 渲染成可拖拽的交互式 HTML。
    中文字体:node['font'] = {'size':14, 'face':'Microsoft YaHei'}
    """
    # notebook=False 去掉 jupyter 特定嵌入
    net = Network(
        height="800px", width="100%",
        bgcolor="#ffffff", font_color="#222222",
        notebook=False, directed=False,
        cdn_resources="in_line",   # 脚本内联,不依赖外网
    )

    # 关掉按钮面板,压文件大小
    # (若想开物理调参面板,改成 net.show_buttons(filter_=['physics']))
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -12000,
          "springLength": 160,
          "springConstant": 0.03,
          "damping": 0.6
        },
        "minVelocity": 0.75
      },
      "interaction": {
        "hover": true,
        "navigationButtons": false,
        "tooltipDelay": 120
      }
    }
    """)

    # 社区 id 映射颜色
    all_cids = sorted(set(word2cid.values()))
    cid2color = {cid: COMMUNITY_PALETTE[i % len(COMMUNITY_PALETTE)]
                 for i, cid in enumerate(all_cids)}

    deg_map = dict(zip(centrality_df["word"], centrality_df["degree"]))

    for n in G.nodes():
        freq = int(overall_freq.get(n, 0))
        # pyvis 的 value 会自动映射节点大小,所以用 log(freq) 做尺度
        val = float(np.log1p(freq) * 10 + 5)
        cid = word2cid.get(n, 0)
        tooltip = (
            f"{n}\n"
            f"整体词频: {freq}\n"
            f"度中心性: {deg_map.get(n, 0):.3f}\n"
            f"社区: #{cid}"
        )
        net.add_node(
            n, label=n, title=tooltip,
            value=val,
            color=cid2color[cid],
            font={"size": 14, "face": "Microsoft YaHei", "color": "#222222"},
            borderWidth=1, shape="dot",
        )

    for u, v, d in G.edges(data=True):
        w = int(d.get("weight", 1))
        net.add_edge(
            u, v,
            value=w,               # 映射边粗细
            title=f"共现 {w} 次",
            color={"color": "#9e9e9e", "opacity": 0.55},
        )

    # 标题注入 HTML <title> 和页面顶部 h3
    net.heading = title

    # save_graph 不会尝试打开浏览器,适合批处理
    # 但 pyvis 0.3.2 在 Windows 下 write_html 用系统默认编码(GBK),
    # 遇到 © 等字符会崩。所以改为:自己取 html 字符串,用 UTF-8 写出。
    save_path = str(save_path)
    html = net.generate_html(notebook=False)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html)

    # 大小检查:超过 1MB 提示(spec Req 6.6 允许为可读性超出)
    size_kb = os.path.getsize(save_path) / 1024
    if size_kb > 1024:
        print(f"  [NOTE] {os.path.basename(save_path)} = {size_kb:.0f} KB "
              f"超过 1MB 目标,但保留节点/悬浮提示完整性,属预期内")


# ============================================================
# 八、国内 vs 国际对比图
# ============================================================
def compare_networks(G_dom, G_intl, overall_freq, word2cid, pos, save_path):
    """
    左右双子图。节点坐标 pos 来自整体网络,两侧共享,只有边粗细各自不同。
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=150)
    for ax, G, subtitle in [
        (axes[0], G_dom, "国内游客视角"),
        (axes[1], G_intl, "国际游客视角"),
    ]:
        nodes = list(G.nodes())
        sizes = _node_size_by_freq(nodes, overall_freq)
        colors, _ = _community_colors(nodes, word2cid)
        widths = _edge_widths(G)

        nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=widths, edge_color="#9e9e9e", alpha=0.55,
        )
        nx.draw_networkx_nodes(
            G, pos, ax=ax, nodelist=nodes,
            node_size=sizes, node_color=colors,
            edgecolors="white", linewidths=1.2, alpha=0.95,
        )
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_color="black")
        ax.set_title(subtitle, fontsize=14, pad=10)
        ax.axis("off")

    fig.suptitle("成都大熊猫基地评论社会语义网络 — 国内 vs 国际对比", fontsize=16, y=0.98)

    # 对比专用说明:此处社区着色来自整体网络,便于左右两图做同色系对比;
    # 这与 network_static_domestic.png / network_static_intl.png 单独图中
    # 的"各组内独立社区"着色不同,阅读时请按图注为准。
    note = ("节点位置:两图一致(来自整体网络的 spring_layout, seed={s})\n"
            "节点大小:整体词频 (sqrt 缩放,两图一致)\n"
            "节点颜色:基于**整体网络** Louvain 社区(两图一致,便于对比)\n"
            "边粗细:反映各自组内的共现强度(两图可不同)").format(s=LAYOUT_SEED)
    fig.text(0.5, 0.02, note, ha="center", fontsize=9, color="#555555")

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_centrality_comparison(cent_dom, cent_intl, save_path):
    """
    国内 Top10 / 国际 Top10 并列。
    列:rank, 国内词, 国内度中心性, 国际词, 国际度中心性
    """
    d_top = cent_dom.nlargest(10, "degree").reset_index(drop=True)
    i_top = cent_intl.nlargest(10, "degree").reset_index(drop=True)
    out = pd.DataFrame({
        "rank": np.arange(1, 11),
        "国内词": d_top["word"],
        "国内度中心性": d_top["degree"].round(3),
        "国际词": i_top["word"],
        "国际度中心性": i_top["degree"].round(3),
    })
    out.to_csv(save_path, index=False, encoding="utf-8-sig")


# ============================================================
# 九、社区分析报告
# ============================================================
def _group_by_community(word2cid, cent_df):
    """把 word -> cid 反转成 cid -> [按度中心性排好序的词列表]。"""
    deg_map = dict(zip(cent_df["word"], cent_df["degree"]))
    buckets = {}
    for w, cid in word2cid.items():
        buckets.setdefault(cid, []).append(w)
    # 每个社区内部按度中心性倒序
    for cid in buckets:
        buckets[cid].sort(key=lambda w: deg_map.get(w, 0), reverse=True)
    return dict(sorted(buckets.items()))


def _infer_theme(words):
    """按关键词模糊归类一个主题名,帮作者减轻负担;找不到就给中性标签。

    节点词经过特殊符号正则过滤,全部为中文,因此这里只用中文关键字匹配。
    新增的"公园/护照/班车/参观"等词是 Trip.com 机翻在中文侧留下的痕迹,
    单列为"国际语境类"一档。
    """
    s = "".join(words[:10])
    if any(k in s for k in ["可爱", "呆萌", "活泼", "睡觉", "团子"]):
        return "动物特征类"
    if any(k in s for k in ["孩子", "家人", "亲子", "小朋友"]):
        return "亲子家庭类"
    if any(k in s for k in ["排队", "人多", "门票", "预约", "交通", "地铁"]):
        return "游览体验类"
    if any(k in s for k in ["环境", "景区", "景点", "园区", "竹子"]):
        return "场景环境类"
    if any(k in s for k in ["推荐", "值得", "不错", "建议", "喜欢"]):
        return "情感态度类"
    # 翻译腔/国际语境:公园(=park)、护照、班车(=shuttle)、参观(=visit)等
    if any(k in s for k in ["公园", "护照", "班车", "巴士", "参观", "访问"]):
        return "国际语境类"
    return "综合主题类"


def write_community_report(all_data, thresholds, save_path):
    """
    all_data = {
        'all':      (word2cid, cent_df, review_count),
        'domestic': (word2cid, cent_df, review_count),
        'intl':     (word2cid, cent_df, review_count),
    }
    thresholds = {'all': int, 'domestic': int, 'intl': int}
    """
    lines = []
    lines.append("# 阶段2:社会语义网络社区分析报告")
    lines.append("")
    lines.append(f"- 节点数:{TOP_N}(整体 Top{TOP_N} 高频词,三组统一节点集)")
    lines.append(f"- 边过滤阈值:整体网络固定 {thresholds['all']};"
                 f"分组网络动态(P{EDGE_PERCENTILE}):"
                 f"国内={thresholds['domestic']},国际={thresholds['intl']}")
    lines.append(f"- 社区检测:Louvain(networkx,seed={LOUVAIN_SEED})")
    lines.append("")

    label2zh = {"all": "整体网络", "domestic": "国内游客网络", "intl": "国际游客网络"}

    # 每组分节
    for key in ["all", "domestic", "intl"]:
        word2cid, cent_df, n_reviews = all_data[key]
        buckets = _group_by_community(word2cid, cent_df)
        n_comm = len(buckets)
        # 区分"有效社区"(≥2 成员)和"孤立节点"(1 成员,度为 0)
        effective = {cid: ws for cid, ws in buckets.items() if len(ws) >= 2}
        isolates = [ws[0] for cid, ws in buckets.items() if len(ws) == 1]
        n_eff = len(effective)
        lines.append(f"## {label2zh[key]}")
        lines.append("")
        lines.append(f"- 评论数:{n_reviews} 条")
        lines.append(f"- 识别到 **{n_eff} 个有效社区**"
                     + (f"(另有 {len(isolates)} 个孤立节点:{', '.join(isolates)})"
                        if isolates else ""))
        if n_eff < 3 or n_eff > 5:
            lines.append(f"- ⚠️ 有效社区数落在推荐区间 3–5 之外。"
                         f"建议检查 `EDGE_THRESHOLD`(当前 {EDGE_THRESHOLD})"
                         f"或 `TOP_N`(当前 {TOP_N})。")
        lines.append("")
        for cid, words in effective.items():
            core = words[0] if words else "—"
            theme = _infer_theme(words)
            lines.append(f"### 社区 #{cid}:{theme}")
            lines.append(f"- 核心词(按度中心性):**{core}**")
            lines.append(f"- 成员词({len(words)}):{'、'.join(words)}")
            lines.append("")

    # 国内 vs 国际差异解读(段落式)
    dom_w2c, dom_cent, _ = all_data["domestic"]
    int_w2c, int_cent, _ = all_data["intl"]
    dom_top5 = set(dom_cent.nlargest(5, "degree")["word"])
    int_top5 = set(int_cent.nlargest(5, "degree")["word"])
    common = dom_top5 & int_top5
    only_dom = dom_top5 - int_top5
    only_int = int_top5 - dom_top5

    lines.append("## 国内 vs 国际社区结构差异解读")
    lines.append("")
    lines.append(
        f"把两组网络并列看,最直观的差异是**核心词重心**。"
        f"国内游客网络和国际游客网络的 Top5 度中心词里,"
        f"共同进入两组核心圈的有 **{len(common)} 个**"
        f"({'、'.join(sorted(common)) if common else '无'})。"
        f"国内独有的高中心度词是 {('、'.join(sorted(only_dom)) if only_dom else '无')},"
        f"国际独有的则是 {('、'.join(sorted(only_int)) if only_int else '无')}。"
        f"这些差异往往对应不同视角下的关注点,"
        f"国内侧更可能谈论本地化的游园体验(排队、带娃、交通),"
        f"国际侧更可能聚焦于大熊猫本身及中国游览符号(panda/chengdu)。"
    )
    lines.append("")
    lines.append(
        f"**社区数对比**:国内 {len(_group_by_community(dom_w2c, dom_cent))} 个,"
        f"国际 {len(_group_by_community(int_w2c, int_cent))} 个。"
        f"数量本身并不代表谁更丰富,但若两边的社区主题切分不同,"
        f"就说明两组游客头脑中"
        f"的概念分块方式确有差异——这是对比研究最有意义的发现。"
        f"具体差异建议结合上文列出的成员词清单人工解读,"
        f"因为自动模糊归类的主题只是给作者一个起点。"
    )
    lines.append("")
    lines.append(
        "**跨组高频但低中心度的词**也值得注意:有些词在两组词频里都高,"
        "但如果在网络里只和少数几个词共现,说明它虽然被经常提到,"
        "却是一个相对'孤立'的话题,没有融入到游客的主要叙事结构中。"
    )
    lines.append("")

    save_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 十、主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="阶段2:社会语义网络分析")
    parser.add_argument("--trial", action="store_true",
                        help="试跑模式:只产出整体静态网络图 network_static_all.png,"
                             "用于先确认字体/布局/社区着色再跑全量")
    args = parser.parse_args()

    set_chinese_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 加载数据 + 分词(共用 text_utils) ----------
    print("[A] 加载数据 & 分词")

    # 数据文件检查(spec Req 1.4)
    if not Path(DATA_PATH).exists():
        print(f"[ERROR] {Path(DATA_PATH).resolve()}", file=sys.stderr)
        sys.exit(1)

    stopwords = load_stopwords(STOPWORDS_PATH)
    init_jieba(CUSTOM_DICT)
    # 阶段2(社会语义网络):使用完整 2000 条样本,不过滤机翻。
    df = load_data(DATA_PATH, filter_translated=False)
    print(f"  🧾 原始样本 {len(df)} 条 (国内 "
          f"{int((df['group']=='国内游客').sum())} / 国际 "
          f"{int((df['group']=='国际游客').sum())})")

    # 列检查(spec Req 1.5)
    required_cols = {"content_clean", "group"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[ERROR] 缺少列: {sorted(missing)}", file=sys.stderr)
        sys.exit(1)

    # 尝试从阶段1 的缓存加载分词结果(省 ~3s)
    from text_utils import load_tokens_cache
    cached = load_tokens_cache(DATA_PATH)
    if cached is not None and set(cached["commentId"]) == set(df["commentId"]):
        df = df.merge(cached[["commentId", "words", "pos_pairs"]], on="commentId", how="left")
    else:
        df = tokenize_dataframe(df, stopwords)
    print(f"  共 {len(df)} 条评论,已分词")

    # ---------- 2. 选 Top30 节点 ----------
    print(f"[B] 构造统一节点集(整体 Top{TOP_N})")
    c_all = count_words(df["words"])
    top_pairs = c_all.most_common(TOP_N)
    node_words = [w for w, _ in top_pairs]
    overall_freq = {w: c for w, c in c_all.items()}   # 节点大小用得到完整词频
    print(f"  Top{TOP_N} 节点:{' / '.join(node_words[:10])} ...")

    # ---------- 3. 共现矩阵 ----------
    print("[C] 共现矩阵(窗口=整条评论)")
    M_all = build_cooccurrence_matrix(df["words"], node_words)
    M_all.to_csv(OUT_DIR / "cooccurrence_matrix_all.csv", encoding="utf-8-sig")
    print(f"  💾 cooccurrence_matrix_all.csv")

    G_all = build_graph_from_matrix(M_all, EDGE_THRESHOLD)
    print(f"  图(整体): {G_all.number_of_nodes()} 节点 / {G_all.number_of_edges()} 边")

    # ---------- 4. 统一布局 + 统一社区着色(基于整体网络) ----------
    print("[D] 统一布局 + Louvain 社区检测(基于整体网络)")
    pos = nx.spring_layout(G_all, seed=LAYOUT_SEED, k=LAYOUT_K,
                           iterations=LAYOUT_ITERS, weight="weight")
    word2cid_all = detect_communities(G_all, seed=LOUVAIN_SEED)
    n_comm = len(set(word2cid_all.values()))
    print(f"  整体网络社区数: {n_comm}")

    # ---------- 5. 中心度(整体) ----------
    cent_all = compute_centrality(G_all, group_label="整体")
    cent_all.to_csv(OUT_DIR / "centrality_all.csv", index=False, encoding="utf-8-sig")

    # ---------- 6. 静态图(整体)—— trial 模式到这里就结束 ----------
    print("[E] 静态网络图(整体)")
    plot_static_network(
        G_all, overall_freq, word2cid_all, pos,
        title="成都大熊猫基地评论社会语义网络 — 整体",
        save_path=OUT_DIR / "network_static_all.png",
    )
    print(f"  🖼 {OUT_DIR / 'network_static_all.png'}")

    if args.trial:
        print("\n✅ 试跑完成。请检查 network_static_all.png:")
        print("   - 中文是否正常(应显示完整汉字,不应有方块)")
        print("   - 节点大小是否有明显差异(高频词更大)")
        print("   - 边粗细是否有明显差异(强共现更粗)")
        print("   - 社区颜色是否明显(2–5 个色块)")
        print("   - 整体是否不乱不挤")
        print("   满意后去掉 --trial 再跑一次,产出全部文件。")
        return

    # ---------- 7. 分组网络 ----------
    print("\n[F] 分组网络(国内 / 国际)")
    df_dom = df[df["group"] == "国内游客"]
    df_intl = df[df["group"] == "国际游客"]
    print(f"  国内 {len(df_dom)} 条 / 国际 {len(df_intl)} 条")

    M_dom = build_cooccurrence_matrix(df_dom["words"], node_words)
    M_intl = build_cooccurrence_matrix(df_intl["words"], node_words)
    M_dom.to_csv(OUT_DIR / "cooccurrence_matrix_domestic.csv", encoding="utf-8-sig")
    M_intl.to_csv(OUT_DIR / "cooccurrence_matrix_intl.csv", encoding="utf-8-sig")
    print(f"  💾 cooccurrence_matrix_domestic.csv / cooccurrence_matrix_intl.csv")

    thr_dom, nz_dom = dynamic_edge_threshold(M_dom)
    thr_intl, nz_intl = dynamic_edge_threshold(M_intl)
    print(f"  动态阈值(国内): P{EDGE_PERCENTILE}={thr_dom}  (非零边 {nz_dom} 条)")
    print(f"  动态阈值(国际): P{EDGE_PERCENTILE}={thr_intl}  (非零边 {nz_intl} 条)")
    G_dom = build_graph_from_matrix(M_dom, thr_dom)
    G_intl = build_graph_from_matrix(M_intl, thr_intl)
    print(f"  图(国内): {G_dom.number_of_nodes()} 节点 / {G_dom.number_of_edges()} 边")
    print(f"  图(国际): {G_intl.number_of_nodes()} 节点 / {G_intl.number_of_edges()} 边")

    # ---------- 8. 分组中心度 ----------
    print("\n[G] 中心度计算")
    cent_dom = compute_centrality(G_dom, group_label="国内")
    cent_intl = compute_centrality(G_intl, group_label="国际")
    cent_dom.to_csv(OUT_DIR / "centrality_domestic.csv", index=False, encoding="utf-8-sig")
    cent_intl.to_csv(OUT_DIR / "centrality_intl.csv", index=False, encoding="utf-8-sig")
    print(f"  💾 centrality_all.csv / centrality_domestic.csv / centrality_intl.csv")

    # ---------- 9. 分组社区(用各自社区着色) ----------
    word2cid_dom = detect_communities(G_dom)
    word2cid_intl = detect_communities(G_intl)

    # ---------- 10. 分组静态图 ----------
    print("\n[H] 分组静态网络图")
    plot_static_network(
        G_dom, overall_freq, word2cid_dom, pos,
        title="成都大熊猫基地评论社会语义网络 — 国内游客视角",
        save_path=OUT_DIR / "network_static_domestic.png",
        edge_threshold=thr_dom,
    )
    plot_static_network(
        G_intl, overall_freq, word2cid_intl, pos,
        title="成都大熊猫基地评论社会语义网络 — 国际游客视角",
        save_path=OUT_DIR / "network_static_intl.png",
        edge_threshold=thr_intl,
    )
    print(f"  🖼 network_static_domestic.png / network_static_intl.png")

    # ---------- 11. 交互 HTML ----------
    print("\n[I] 交互式网络图(pyvis)")
    plot_interactive_network(
        G_all, overall_freq, word2cid_all, cent_all,
        title="社会语义网络(整体)",
        save_path=OUT_DIR / "network_interactive_all.html",
    )
    plot_interactive_network(
        G_dom, overall_freq, word2cid_dom, cent_dom,
        title="社会语义网络(国内游客)",
        save_path=OUT_DIR / "network_interactive_domestic.html",
    )
    plot_interactive_network(
        G_intl, overall_freq, word2cid_intl, cent_intl,
        title="社会语义网络(国际游客)",
        save_path=OUT_DIR / "network_interactive_intl.html",
    )
    print(f"  🌐 network_interactive_{{all,domestic,intl}}.html")

    # ---------- 12. 对比图 + 对比 CSV ----------
    print("\n[J] 国内 vs 国际对比")
    # 对比图为了视觉一致,用整体网络的社区着色
    compare_networks(
        G_dom, G_intl, overall_freq, word2cid_all, pos,
        save_path=OUT_DIR / "network_comparison.png",
    )
    print(f"  🖼 network_comparison.png")

    write_centrality_comparison(
        cent_dom, cent_intl,
        save_path=OUT_DIR / "centrality_comparison.csv",
    )
    print(f"  💾 centrality_comparison.csv")

    # ---------- 13. 社区分析 md ----------
    print("\n[K] 社区分析报告")
    write_community_report(
        all_data={
            "all":      (word2cid_all,  cent_all,  len(df)),
            "domestic": (word2cid_dom,  cent_dom,  len(df_dom)),
            "intl":     (word2cid_intl, cent_intl, len(df_intl)),
        },
        thresholds={"all": EDGE_THRESHOLD, "domestic": thr_dom, "intl": thr_intl},
        save_path=OUT_DIR / "community_analysis.md",
    )
    print(f"  📝 community_analysis.md")

    # ---------- 14. 产出清单 ----------
    print("\n" + "=" * 60)
    print(f"✅ 阶段2 全部产出完成,位于 {OUT_DIR}/")
    for p in sorted(OUT_DIR.iterdir()):
        if p.is_file():
            size_kb = p.stat().st_size / 1024
            print(f"  {p.name:<42}  {size_kb:>8.1f} KB")


if __name__ == "__main__":
    main()
