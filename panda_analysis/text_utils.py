# -*- coding: utf-8 -*-
"""
text_utils.py
==============
跨阶段复用的文本处理工具:停用词、jieba 初始化、分词、词频、数据加载。

背景
----
原先这些函数放在 `01_word_freq.py` 里,`02_network.py` 通过 importlib
把 "01 开头的数字文件" 动态加载进来(因为 `import 01_word_freq` 语法不合法)。
这种做法脆弱、难以维护,且容易让"阶段1 默认过滤机翻"的副作用悄悄传染到
阶段2。

现在把工具函数集中到本模块,三阶段脚本都 `from text_utils import xxx`,
每个调用方显式声明自己要不要过滤机翻,样本基数清晰可见。
"""

import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import jieba
import jieba.posseg as pseg
import pandas as pd


# ============================================================
# 常量(与阶段1 保持一致)
# ============================================================
# 分词长度区间
MIN_LEN = 2
MAX_LEN = 8
# 词性白名单:名词族(n*)、动词(v)、形容词(a)
KEEP_NOUN_PREFIX = ("n",)
KEEP_OTHER = ("v", "a")

# 预编译正则(模块级)
_PURE_DIGIT = re.compile(r"^\d+$")
_HAS_SPECIAL = re.compile(r"[^一-龥a-zA-Z0-9]")

# 阶段3 产出路径,load_data 用来可选地过滤机翻样本
SENTIMENT_RESULTS_PATH = Path("output/3_sentiment/sentiment_results.csv")


# ============================================================
# 词表加载
# ============================================================
def load_stopwords(path: str) -> set:
    """按行读取停用词表,空行跳过,返回去重 set。"""
    s = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                s.add(w)
    return s


def init_jieba(custom_dict: str) -> None:
    """加载 jieba 自定义词典,保护专有词不被切碎。"""
    jieba.load_userdict(custom_dict)


# ============================================================
# 数据加载
# ============================================================
def _comment_id_set(series) -> set:
    """把 commentId 列规范化为 int 集合,用于跨产物一致性校验。"""
    return set(pd.to_numeric(series, errors="coerce").dropna().astype("int64"))


def load_data(csv_path: str, limit: int | None = None,
              filter_translated: bool = False) -> pd.DataFrame:
    """读 CSV(UTF-8-sig),可选地合并阶段3 的机翻标签并剔除机翻样本。

    filter_translated 语义:
        - 阶段1(词频/差异词):传 True —— 机翻"护照/班车/红色"等伪影会主导词频;
        - 阶段2/3/4:保持 False —— 用完整 2000 条,避免国际组稀疏。

    此前默认 True 会让 01/02 悄悄变成 1347 条,而 03/04 用 2000 条,
    同一份报告里基数切换。现默认 False,各调用方显式传参。
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if filter_translated and SENTIMENT_RESULTS_PATH.exists():
        sent = pd.read_csv(SENTIMENT_RESULTS_PATH, encoding="utf-8-sig",
                           usecols=["commentId", "is_translated"])
        df_ids = _comment_id_set(df["commentId"])
        sent_ids = _comment_id_set(sent["commentId"])
        if df_ids != sent_ids:
            missing = len(df_ids - sent_ids)
            extra = len(sent_ids - df_ids)
            print("⚠️  sentiment_results.csv 与当前样本 commentId 不一致,"
                  f"跳过机翻过滤 (缺 {missing} 条,多 {extra} 条)。"
                  "请重跑 03_sentiment.py 后再过滤。")
        else:
            # 规范化为纯 bool(CSV 读入后可能混合 bool / "True" / "true")
            sent["is_translated"] = sent["is_translated"].astype(str).str.lower().isin(
                ["true", "1", "yes"])
            df = df.merge(sent, on="commentId", how="left")
            coverage = df["is_translated"].notna().sum() / max(len(df), 1)
            # 一致性已校验,merge 后 NaN 只可能来自异常 ID 类型;保守视为非机翻
            mask_tr = df["is_translated"].fillna(False).astype(bool)
            n_tr = int(mask_tr.sum())
            before = len(df)
            df = df.loc[~mask_tr].drop(columns=["is_translated"]).copy()
            print(f"🧹 过滤疑似机翻评论: 剔除 {n_tr} 条 ({before} → {len(df)}); "
                  f"覆盖率 {coverage:.1%}")
    elif filter_translated:
        print("⚠️  未找到 output/3_sentiment/sentiment_results.csv,跳过机翻过滤")

    if limit is not None:
        df = df.head(limit).copy()
    return df


# ============================================================
# 分词
# ============================================================
def tokenize_with_pos(text: str, stopwords: set) -> list:
    """对单条评论分词,返回 [(word, pos), ...]。

    过滤任一:
        - 词长 [MIN_LEN, MAX_LEN] 之外
        - 在停用词中
        - 纯数字
        - 含特殊符号(非中文/英文/数字)
        - 词性不在 n*/v/a 白名单中
    """
    if not isinstance(text, str) or not text:
        return []
    out = []
    for w, flag in pseg.cut(text):
        w = w.strip()
        if not w:
            continue
        if len(w) < MIN_LEN or len(w) > MAX_LEN:
            continue
        if w in stopwords:
            continue
        if _PURE_DIGIT.match(w):
            continue
        if _HAS_SPECIAL.search(w):
            continue
        if flag.startswith(KEEP_NOUN_PREFIX) or flag in KEEP_OTHER:
            out.append((w, flag))
    return out


def tokenize_dataframe(df: pd.DataFrame, stopwords: set,
                       text_col: str = "content_clean") -> pd.DataFrame:
    """对 DataFrame 的文本列做批量分词,返回新增 words / pos_pairs 两列的 df。"""
    pairs_col = []
    words_col = []
    for t in df[text_col].fillna("").astype(str):
        pairs = tokenize_with_pos(t, stopwords)
        pairs_col.append(pairs)
        words_col.append([w for w, _ in pairs])
    df = df.copy()
    df["pos_pairs"] = pairs_col
    df["words"] = words_col
    return df


# ============================================================
# 词频统计
# ============================================================
def count_words(series_of_lists: Iterable) -> Counter:
    """把多条评论的词列表汇成一个 Counter。"""
    c = Counter()
    for lst in series_of_lists:
        c.update(lst)
    return c


# ============================================================
# 分词缓存(可选加速)
# ============================================================
TOKENS_CACHE_PATH = Path("output/1_word_freq/tokens_cache.pkl")


def save_tokens_cache(df: pd.DataFrame) -> None:
    """把分词结果(words + pos_pairs 列)序列化到 pkl,供阶段 2 复用。

    只保存 commentId + words + pos_pairs + group 四列,体积小、加载快。
    """
    cols = ["commentId", "words", "pos_pairs"]
    if "group" in df.columns:
        cols.append("group")
    TOKENS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_pickle(str(TOKENS_CACHE_PATH))
    size_kb = TOKENS_CACHE_PATH.stat().st_size / 1024
    print(f"💾 分词缓存已保存: {TOKENS_CACHE_PATH} ({size_kb:.1f} KB)")


def load_tokens_cache(csv_path: str) -> pd.DataFrame | None:
    """尝试加载分词缓存。

    校验逻辑:
        - pkl 文件存在
        - pkl 的 commentId 集合与当前 CSV 完全一致(防止数据更新后用旧缓存)

    返回 None 表示缓存不可用,调用方应走正常分词流程。
    返回 DataFrame 时已包含 words / pos_pairs / group 列,可直接 merge 回原 df。
    """
    if not TOKENS_CACHE_PATH.exists():
        return None
    try:
        cached = pd.read_pickle(str(TOKENS_CACHE_PATH))
    except Exception:
        return None

    # 校验 commentId 一致性
    df_orig = pd.read_csv(csv_path, encoding="utf-8-sig", usecols=["commentId"])
    if set(cached["commentId"]) != set(df_orig["commentId"]):
        print("⚠️  分词缓存 commentId 与当前数据不一致,将重新分词")
        return None

    print(f"⚡ 从缓存加载分词结果: {TOKENS_CACHE_PATH}")
    return cached
