# -*- coding: utf-8 -*-
"""
03_sentiment.py
================
阶段3:基于 Gemini API 的情感与方面分析

核心能力
--------
1) 基础框架:读取 .env 的 API key、构造 Gemini 2.5 Flash 客户端
2) 限速 + 指数退避重试:每批 BATCH_SIZE 条,批间 sleep SLEEP_BETWEEN_BATCHES 秒
3) 批量 + 断点续传:每 SAVE_EVERY_N 条落一次 sentiment_progress.csv,
   中断再跑会跳过已完成的 commentId 自动接着来
4) 严格 JSON prompt + 解析校验:每条返回
   {comment_id, sentiment, intensity, aspects, key_phrase, is_translated}
5) 双后端:先调 API 拿情感标签,再基于结果产 7 个图表 + 1 份报告

运行示例
--------
    python 03_sentiment.py --trial         # 10 条试跑,打印原始 JSON,调 prompt
    python 03_sentiment.py --validate      # 100 条小批验证,抽查质量
    python 03_sentiment.py                 # 全量 2000 条(断点续传)
    python 03_sentiment.py --analyze-only  # 跳过 API,基于已有结果只出图表

参数调优提示
-----------
- 免费额度(15 RPM, 月度 token cap 较低):BATCH_SIZE=10, SLEEP=4
- 付费额度:BATCH_SIZE=40, SLEEP=1(当前默认)

断点续传说明
-----------
- 中间态:`output/3_sentiment/sentiment_progress.csv`
- 只要该文件存在,再次运行会从"已处理的 commentId"之外接着跑
- 跑完会 merge 回原 2000 条,输出 `sentiment_results.csv`
"""

import argparse
import json
import os
import sys
import time
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# 静默 google-generativeai 的 FutureWarning(该库已弃用但仍可用)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import google.generativeai as genai  # noqa: E402

from font_config import set_chinese_font  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


# ============================================================
# 一、配置
# ============================================================
DATA_PATH = "data/panda_sample_2000.csv"
OUT_DIR = Path("output/3_sentiment")
PROGRESS_FILE = OUT_DIR / "sentiment_progress.csv"
RESULTS_FILE = OUT_DIR / "sentiment_results.csv"
WEIGHTED_DIST_FILE = OUT_DIR / "sentiment_weighted_distribution.csv"
CLEAN_DATA_PATH = Path("../panda_spider/data/panda_reviews_clean.csv")

MODEL_NAME = "gemini-2.5-flash"

BATCH_SIZE = 40            # 每次塞给 Gemini 的评论数(付费额度下可以加大)
SLEEP_BETWEEN_BATCHES = 1  # 秒 —— 付费档 RPM 上限很高,1s 足够安全
MAX_RETRIES = 3            # 单次调用失败重试次数
RETRY_BASE_DELAY = 4       # 指数退避起点(第1次4s,第2次8s,第3次16s)
SAVE_EVERY_N = 40          # 每处理多少条评论落一次盘(和 BATCH_SIZE 对齐)

# 8 个固定方面类别(与 prompt 保持严格一致)
ASPECTS_ALLOWED = {
    "熊猫互动", "环境景观", "服务设施", "票务交通",
    "拥挤排队", "讲解科普", "餐饮购物", "性价比",
}
# 画图时固定顺序,保证跨图一致
ASPECTS_ORDER = ["熊猫互动", "环境景观", "服务设施", "票务交通",
                 "拥挤排队", "讲解科普", "餐饮购物", "性价比"]
SENTIMENTS_ALLOWED = {"正面", "负面", "中性"}
INTENSITIES_ALLOWED = {"强", "中", "弱"}

# 情感配色(与 spec 一致)
SENT_COLORS = {
    "正面": "#27AE60",
    "负面": "#E74C3C",
    "中性": "#95A5A6",
    "error": "#000000",
}
GROUP_COLORS = {
    "国内游客": "#E74C3C",
    "国际游客": "#1ABC9C",
}

# 与 panda_spider/sample_for_analysis.py 保持一致。用于把清洗后总体数据
# 回填 group,再按 group × score 计算分层样本的总体校正权重。
FOREIGN_COUNTRIES = {
    "新加坡", "日本", "马来西亚", "泰国", "美国", "澳大利亚",
    "英国", "德国", "荷兰", "韩国", "瑞士", "奥地利",
    "尼泊尔", "斯里兰卡", "加拿大",
}


# ============================================================
# 数据一致性校验
# ============================================================
def _comment_id_set(df: pd.DataFrame) -> set[int]:
    """把 DataFrame 的 commentId 列规范化为 int 集合。"""
    return set(pd.to_numeric(df["commentId"], errors="coerce")
               .dropna().astype("int64"))


def _id_preview(ids: set[int], n: int = 5) -> str:
    return ", ".join(str(x) for x in sorted(ids)[:n])


def validate_comment_ids(result_df: pd.DataFrame, sample_df: pd.DataFrame,
                         label: str) -> bool:
    """确认分析结果和当前样本的 commentId 完全一致。"""
    sample_ids = _comment_id_set(sample_df)
    result_ids = _comment_id_set(result_df)
    missing = sample_ids - result_ids
    extra = result_ids - sample_ids
    if not missing and not extra:
        return True

    print(f"[ERROR] {label} 与当前 {DATA_PATH} 的 commentId 不一致。", file=sys.stderr)
    print(f"        当前样本 {len(sample_ids)} 条;结果 {len(result_ids)} 条;"
          f"缺 {len(missing)} 条;多 {len(extra)} 条。", file=sys.stderr)
    if missing:
        print(f"        样本中缺失结果的 commentId 示例:{_id_preview(missing)}",
              file=sys.stderr)
    if extra:
        print(f"        结果中不属于当前样本的 commentId 示例:{_id_preview(extra)}",
              file=sys.stderr)
    print("        请重跑 03_sentiment.py,或先备份/删除旧的阶段3结果。",
          file=sys.stderr)
    return False


def validate_sentiment_complete(result_df: pd.DataFrame, label: str) -> bool:
    """确认结果文件已经包含可用的 sentiment 列且没有空值。"""
    if "sentiment" not in result_df.columns:
        print(f"[ERROR] {label} 缺少 sentiment 列。", file=sys.stderr)
        return False
    missing = int(result_df["sentiment"].isna().sum())
    if missing:
        print(f"[ERROR] {label} 有 {missing} 条缺少 sentiment,"
              "不能用于 --analyze-only。", file=sys.stderr)
        return False
    return True


# ============================================================
# 二、Prompt 模板(严格按 spec 要求)
# ============================================================
PROMPT_TEMPLATE = """你是一名旅游学领域的专业研究员,正在对成都大熊猫繁育研究基地的游客评论做情感和方面分析。

请仔细阅读下面{n}条评论,对每条做以下判断,以严格的JSON数组格式返回:

【评论列表】
{comments}

【输出要求】
返回一个JSON数组,数组中每个对象对应一条评论,顺序保持一致。每个对象包含以下字段:

1. "comment_id": 原始comment_id(整数)

2. "sentiment": 必须从["正面", "负面", "中性"]中选一个
   - 正面:游客整体表达赞美、推荐、满意
   - 负面:游客整体表达不满、失望、吐槽、抱怨
   - 中性:陈述事实、混合情感、无明显倾向

3. "intensity": 必须从["强", "中", "弱"]中选一个
   - 强:情感非常明确、用词强烈(超棒/绝对推荐/极差/避雷)
   - 中:情感清晰但用词平和(不错/还行/一般/失望)
   - 弱:情感隐含、需要推断(例如只描述事实但暗含态度)

4. "aspects": 评论涉及的方面,从以下8类中选,可多选,数组格式:
   ["熊猫互动", "环境景观", "服务设施", "票务交通", "拥挤排队", "讲解科普", "餐饮购物", "性价比"]
   - 不涉及任何方面就返回空数组 []

5. "key_phrase": 最能代表该评论情感的关键短语,从原文摘出,不超过15字

6. "is_translated": 是否疑似机器翻译,布尔值
   - 翻译特征:语句生硬、用词不符合中文习惯、有"它/他/她"指熊猫等指代错误、句式直译

【重要原则】
- sentiment和intensity必须给出,不能"无法判断"
- aspects只从8个固定类别中选,不要自创
- 严格输出JSON数组,不要任何额外说明文字、不要markdown代码块
- 第一个字符必须是 [,最后一个字符必须是 ]

请开始分析:"""


# ============================================================
# 三、基础工具
# ============================================================
def init_client():
    """读取 .env 并初始化 Gemini 客户端,返回 GenerativeModel 实例。"""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("[ERROR] 找不到 GEMINI_API_KEY,请在 .env 填好后重试。", file=sys.stderr)
        sys.exit(1)
    genai.configure(api_key=api_key)
    # 温度低一点,让情感判断更稳定;强制 JSON 输出
    model = genai.GenerativeModel(
        MODEL_NAME,
        generation_config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "max_output_tokens": 16384,
        },
    )
    return model


def build_prompt(batch_rows):
    """把一批 DataFrame 行拼成 prompt 里的"评论列表"文本。
    每条形如:
        <<ID=12345 SCORE=4.0>> 这只熊猫太可爱了……

    使用 << >> 作为分隔符(而非 [ ]),因为评论原文里方括号常见,
    容易和系统标记混淆;而 << >> 在中文评论里几乎不出现。
    """
    lines = []
    for _, row in batch_rows.iterrows():
        cid = int(row["commentId"])
        content = str(row["content_clean"]).strip().replace("\n", " ")
        # 防 prompt 注入:去掉评论里的 ``` 和 << >> 控制符
        content = (content.replace("```", "")
                   .replace("<<", "").replace(">>", "")
                   .replace("【", "(").replace("】", ")"))
        score = row.get("score", "")
        lines.append(f"<<ID={cid} SCORE={score}>> {content}")
    return PROMPT_TEMPLATE.format(n=len(batch_rows), comments="\n".join(lines))


def _strip_code_fence(text):
    """若模型仍在输出里包 ```json ... ``` ,剥掉。"""
    text = text.strip()
    if text.startswith("```"):
        # 去掉第一行的 ```json 或 ``` 以及结尾的 ```
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_and_validate(text, expected_ids):
    """
    把模型返回的文本解析成 list[dict],并做字段合法性校验。
    校验顺序:
        1) JSON 可解析且是数组
        2) 元素数量与预期一致(不一致也继续,但发 warning)
        3) 按 comment_id 反查到原批次对应行;如果模型错位了,本函数会
           尽可能按 id 重排;找不到 id 的条目直接剔除
        4) 字段值不合法时做保底纠正(非法枚举值 -> "中性"/"中")
    返回 (records, warnings);warnings 记录异常但不中断。
    """
    warnings_list = []
    text = _strip_code_fence(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败: {e}; 前200字符: {text[:200]}")

    if not isinstance(data, list):
        raise ValueError(f"返回不是 JSON 数组,而是 {type(data).__name__}")

    if len(data) != len(expected_ids):
        warnings_list.append(
            f"条数不一致:返回 {len(data)} 条,预期 {len(expected_ids)} 条")

    expected_set = set(int(x) for x in expected_ids)
    seen = set()
    cleaned = []

    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            warnings_list.append(f"第{i+1}条不是 dict,已跳过")
            continue

        # comment_id —— 必须是整数,且应当在 expected_set 内
        try:
            cid = int(obj.get("comment_id"))
        except Exception:
            warnings_list.append(f"第{i+1}条 comment_id 不是整数,已跳过")
            continue

        if cid not in expected_set:
            warnings_list.append(
                f"comment_id={cid} 不在本批预期 id 集中,已跳过")
            continue
        if cid in seen:
            warnings_list.append(f"comment_id={cid} 重复返回,已跳过后者")
            continue
        seen.add(cid)

        # sentiment / intensity —— 不合法就纠到 "中性" / "中"
        sent = obj.get("sentiment", "")
        if sent not in SENTIMENTS_ALLOWED:
            warnings_list.append(f"id={cid} sentiment 非法:{sent!r} -> 中性")
            sent = "中性"
        inten = obj.get("intensity", "")
        if inten not in INTENSITIES_ALLOWED:
            warnings_list.append(f"id={cid} intensity 非法:{inten!r} -> 中")
            inten = "中"

        # aspects —— 过滤到固定 8 类;不在其中的剔除(用 | 作为分隔符避免与原文逗号冲突)
        raw_aspects = obj.get("aspects", []) or []
        if not isinstance(raw_aspects, list):
            raw_aspects = []
        aspects = []
        for a in raw_aspects:
            if isinstance(a, str) and a in ASPECTS_ALLOWED and a not in aspects:
                aspects.append(a)
            elif isinstance(a, str):
                warnings_list.append(f"id={cid} aspect 越界剔除:{a!r}")

        # key_phrase —— 截断到 30 字(防止模型过长)
        kp = str(obj.get("key_phrase", "")).strip().replace("\n", " ")
        if len(kp) > 30:
            kp = kp[:30]

        # is_translated
        tr = obj.get("is_translated", False)
        if not isinstance(tr, bool):
            tr = bool(tr)

        cleaned.append({
            "commentId": cid,
            "sentiment": sent,
            "intensity": inten,
            "aspects": "|".join(aspects),  # 用 | 分隔,避免和评论里的逗号混淆
            "key_phrase": kp,
            "is_translated": tr,
        })

    # 检查漏掉的 id
    missing = expected_set - seen
    if missing:
        warnings_list.append(
            f"本批有 {len(missing)} 个 id 未返回(样例: {list(missing)[:3]})")

    return cleaned, warnings_list


# ============================================================
# 四、调用一批(含重试)
# ============================================================
def call_one_batch(model, batch_rows, trial_mode=False, usage_stats=None):
    """
    调用 Gemini 处理一批评论,带指数退避重试。
    trial_mode=True 时,把原始响应原样打印,便于人工肉眼调 prompt。
    usage_stats: 可选 dict,若传入,将累加 prompt/candidates/total token 数。
    返回:已解析并过滤的 list[dict]。
    """
    prompt = build_prompt(batch_rows)
    expected_ids = [int(x) for x in batch_rows["commentId"].tolist()]

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = model.generate_content(prompt)
            text = resp.text if hasattr(resp, "text") else str(resp)

            # 累计 token(若 API 返回)
            if usage_stats is not None:
                try:
                    um = getattr(resp, "usage_metadata", None)
                    if um is not None:
                        usage_stats["prompt_tokens"] += getattr(um, "prompt_token_count", 0) or 0
                        usage_stats["output_tokens"] += getattr(um, "candidates_token_count", 0) or 0
                        usage_stats["total_tokens"] += getattr(um, "total_token_count", 0) or 0
                except Exception:
                    pass

            if trial_mode:
                print("\n" + "=" * 70)
                print(f"📨 Gemini 原始响应(第{attempt}次尝试):")
                print("=" * 70)
                print(text)
                print("=" * 70)

            records, warns = parse_and_validate(text, expected_ids)
            for w in warns:
                print(f"  [WARN] {w}")
            return records

        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(f"  [RETRY] 第{attempt}次失败({e.__class__.__name__}: {e}),"
                      f"{delay}s 后重试")
                time.sleep(delay)
            else:
                print(f"  [FAIL] 重试 {MAX_RETRIES} 次均失败:{e}")

    # 3 次都失败:为这一批每条生成一个 error 占位
    return [{
        "commentId": cid,
        "sentiment": "error",
        "intensity": "error",
        "aspects": "",
        "key_phrase": f"ERROR: {last_err}"[:30],
        "is_translated": False,
    } for cid in expected_ids]


# ============================================================
# 五、批处理 + 断点续传
# ============================================================
def load_progress(valid_ids: set[int] | None = None):
    """读取已有进度;返回 (已处理的 commentId 集合, 已处理的 DataFrame)。
    顺便做一次去重(按 commentId 保留最后一次),防止早期运行的重复追加。"""
    if not PROGRESS_FILE.exists():
        return set(), pd.DataFrame()

    df = pd.read_csv(PROGRESS_FILE, encoding="utf-8-sig")
    if df.empty:
        return set(), df

    before = len(df)
    df["_commentId_num"] = pd.to_numeric(df["commentId"], errors="coerce")
    df = df[df["_commentId_num"].notna()].copy()
    df["commentId"] = df["_commentId_num"].astype("int64")
    df = df.drop(columns=["_commentId_num"])
    df = df.drop_duplicates(subset="commentId", keep="last")
    # 只保留 sentiment 合法的记录作为"已完成"
    df_valid = df[df["sentiment"].isin(SENTIMENTS_ALLOWED)]
    if valid_ids is not None:
        before_filter = len(df_valid)
        df_valid = df_valid[df_valid["commentId"].isin(valid_ids)].copy()
        dropped = before_filter - len(df_valid)
        if dropped:
            print(f"🔁 进度文件含 {dropped} 条旧样本记录,已忽略")

    # 若发现了重复或非法记录,回写一份干净的 progress 文件
    if len(df_valid) != before:
        df_valid.to_csv(PROGRESS_FILE, index=False, encoding="utf-8-sig")
        print(f"🔁 进度文件去重/清理: {before} -> {len(df_valid)}")

    done = set(df_valid["commentId"].astype("int64").tolist())
    print(f"🔁 发现进度文件,已完成 {len(done)} 条")
    return done, df_valid


def append_to_progress(new_records):
    """把一批新结果追加到 sentiment_progress.csv(无则创建)。"""
    df_new = pd.DataFrame(new_records)
    header = not PROGRESS_FILE.exists()
    df_new.to_csv(PROGRESS_FILE, mode="a", index=False,
                  header=header, encoding="utf-8-sig")


def run_batches(model, df_todo, trial_mode=False):
    """
    按 BATCH_SIZE 一批地跑,每 SAVE_EVERY_N 条落一次盘,返回所有新结果列表。
    """
    total = len(df_todo)
    n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    buffer = []        # 未落盘的记录缓存
    all_records = []   # 本次运行的全部记录
    ok_count = 0
    err_count = 0
    usage_stats = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    t_start = time.time()

    for bi in range(n_batches):
        s, e = bi * BATCH_SIZE, min((bi + 1) * BATCH_SIZE, total)
        batch = df_todo.iloc[s:e]

        print(f"\n▶ 批次 {bi+1}/{n_batches}  评论 {s+1}-{e}  (共 {total} 条)")
        records = call_one_batch(model, batch, trial_mode=trial_mode,
                                 usage_stats=usage_stats)
        buffer.extend(records)
        all_records.extend(records)

        this_ok = sum(1 for r in records if r["sentiment"] != "error")
        this_err = len(records) - this_ok
        ok_count += this_ok
        err_count += this_err
        print(f"  ✔ 成功 {this_ok} / 失败 {this_err};累计成功 {ok_count} / 失败 {err_count}")

        # 达到阈值就落盘
        if len(buffer) >= SAVE_EVERY_N:
            append_to_progress(buffer)
            print(f"  💾 已落盘 {len(buffer)} 条到 {PROGRESS_FILE.name}")
            buffer = []

        # trial 模式不睡,其余正常 sleep 限速
        if not trial_mode and bi < n_batches - 1:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    # 残余落盘
    if buffer:
        append_to_progress(buffer)
        print(f"  💾 末尾落盘 {len(buffer)} 条")

    elapsed = time.time() - t_start
    total_done = ok_count + err_count
    print("\n" + "=" * 60)
    print(f"⏱ 用时 {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"📊 成功 {ok_count} / 失败 {err_count} "
          f"/ 成功率 {ok_count/max(total_done,1):.1%}")
    if usage_stats["total_tokens"] > 0:
        print(f"🪙 Token 消耗: prompt={usage_stats['prompt_tokens']:,} / "
              f"output={usage_stats['output_tokens']:,} / "
              f"total={usage_stats['total_tokens']:,}")
        if total_done > 0:
            print(f"   (≈ {usage_stats['total_tokens']/total_done:.0f} tokens / 条)")
        # 估算费用(Gemini 2.5 Flash 2025 定价:input $0.15/M, output $0.60/M)
        cost_input = usage_stats["prompt_tokens"] / 1_000_000 * 0.15
        cost_output = usage_stats["output_tokens"] / 1_000_000 * 0.60
        cost_total = cost_input + cost_output
        print(f"💰 估算费用: ~${cost_total:.4f} "
              f"(input ${cost_input:.4f} + output ${cost_output:.4f})")
    else:
        print("🪙 Token 消耗:API 未返回统计")
    return all_records


# ============================================================
# 六、合并 + 生成最终 CSV
# ============================================================
def finalize_results(df_orig, out_path: Path | None = None):
    """
    读取 sentiment_progress.csv,merge 回原 DataFrame,
    产出 sentiment_results.csv。
    """
    prog = pd.read_csv(PROGRESS_FILE, encoding="utf-8-sig")
    prog["_commentId_num"] = pd.to_numeric(prog["commentId"], errors="coerce")
    prog = prog[prog["_commentId_num"].notna()].copy()
    prog["commentId"] = prog["_commentId_num"].astype("int64")
    prog = prog.drop(columns=["_commentId_num"])
    # 每个 commentId 应当只有一行;如果断点续传跑重了,保留最后一次结果
    prog = prog.drop_duplicates(subset="commentId", keep="last")
    current_ids = _comment_id_set(df_orig)
    prog = prog[prog["commentId"].isin(current_ids)].copy()

    merged = df_orig.merge(prog, on="commentId", how="left")
    missing = int(merged["sentiment"].isna().sum())
    if missing:
        print(f"⚠️  当前样本仍有 {missing} 条没有情感结果;"
              "如果不是 validate 模式,请继续重跑 03_sentiment.py")
    out = out_path or RESULTS_FILE
    merged.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"💾 {out}  ({len(merged)} 行)")
    return merged


# ============================================================
# 七、分析产出:图表 + 报告
# ============================================================
def _split_aspects(cell):
    """把 'a|b|c' 拆回列表;兼容旧数据的逗号分隔;空/NaN -> []"""
    if pd.isna(cell) or not str(cell).strip():
        return []
    s = str(cell)
    # 先按 | 拆;若全部元素都不在 ASPECTS_ALLOWED 内,退回到按逗号拆(兼容早期数据)
    parts = [x.strip() for x in s.split("|") if x.strip()]
    if parts and all(p not in ASPECTS_ALLOWED for p in parts):
        parts = [x.strip() for x in s.split(",") if x.strip()]
    return parts


def _valid_df(df):
    """只保留 sentiment 合法(排除 error 和缺失)的行,用于图表统计。"""
    return df[df["sentiment"].isin(SENTIMENTS_ALLOWED)].copy()


def _tag_group_for_weight(row) -> str:
    """复用采样脚本的分组规则,给清洗后总体数据回填 group。"""
    ft = row.get("fromTypeText")
    ip = row.get("ipLocatedName")
    if ft == "来自Trip.com":
        return "国际游客"
    if isinstance(ip, str) and ip in FOREIGN_COUNTRIES:
        return "国际游客"
    return "国内游客"


def _normalize_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out = out[out["score"].isin([1, 2, 3, 4, 5])].copy()
    out["score"] = out["score"].astype(int)
    return out


def _sentiment_pct_row(label: str, sub: pd.DataFrame,
                       weight_col: str | None = None) -> dict:
    weights = (
        sub[weight_col].astype(float)
        if weight_col and weight_col in sub.columns
        else pd.Series(1.0, index=sub.index)
    )
    total = float(weights.sum())
    row = {"分组": label, "估计总体量": round(total, 1)}
    for sent in ["正面", "中性", "负面"]:
        val = float(weights[sub["sentiment"] == sent].sum())
        row[f"{sent}估计量"] = round(val, 1)
        row[f"{sent}占比"] = round(val / total * 100, 1) if total else 0.0
    return row


def build_weighted_sentiment_distribution(df: pd.DataFrame,
                                          clean_path: Path = CLEAN_DATA_PATH
                                          ) -> pd.DataFrame | None:
    """按清洗后总体的 group × score 分布给分层样本加权。

    当前 2000 条样本是评分分层样本,适合比较低分/高分和国内/国际差异,
    但未加权的整体情感比例不能代表总体口碑。这里用清洗后总体数据中
    每个 group × score 单元格的数量除以样本中同单元格数量作为权重。
    """
    if not clean_path.exists():
        print(f"  [WARN] 未找到清洗总体数据 {clean_path},跳过加权情感统计")
        return None

    d = _normalize_score(_valid_df(df))
    if d.empty:
        print("  [WARN] 没有有效情感样本,跳过加权情感统计")
        return None

    pop = pd.read_csv(clean_path, encoding="utf-8-sig")
    pop = _normalize_score(pop)
    if "group" not in pop.columns:
        pop["group"] = pop.apply(_tag_group_for_weight, axis=1)

    # 与 sample_for_analysis.py 默认口径一致:只估计样本覆盖时段内的总体。
    if "publishDate" in d.columns and "publishDate" in pop.columns:
        dates = pd.to_datetime(d["publishDate"], errors="coerce").dropna()
        if len(dates):
            min_date = dates.min().strftime("%Y-%m-%d")
            before = len(pop)
            pop = pop[pop["publishDate"].astype(str) >= min_date].copy()
            print(f"  加权总体口径: publishDate >= {min_date} "
                  f"({before} -> {len(pop)} 条)")

    strata = ["group", "score"]
    pop_counts = (pop.groupby(strata).size()
                  .rename("population_n").reset_index())
    sample_counts = (d.groupby(strata).size()
                     .rename("sample_n").reset_index())
    weights = pop_counts.merge(sample_counts, on=strata, how="inner")
    weights["weight"] = weights["population_n"] / weights["sample_n"]

    missing = sample_counts.merge(weights[strata], on=strata, how="left",
                                  indicator=True)
    missing = missing[missing["_merge"] == "left_only"]
    if len(missing):
        print("  [WARN] 部分样本分层在总体中无匹配,这些记录将按权重 1 处理")

    d = d.merge(weights[strata + ["population_n", "sample_n", "weight"]],
                on=strata, how="left")
    d["weight"] = d["weight"].fillna(1.0)

    rows = [
        _sentiment_pct_row("整体(按总体评分分布加权)", d, "weight"),
        _sentiment_pct_row("国内游客(加权)", d[d["group"] == "国内游客"], "weight"),
        _sentiment_pct_row("国际游客(加权)", d[d["group"] == "国际游客"], "weight"),
    ]
    out = pd.DataFrame(rows)
    out["样本量"] = [
        len(d),
        int((d["group"] == "国内游客").sum()),
        int((d["group"] == "国际游客").sum()),
    ]
    return out


def write_weighted_sentiment_outputs(df: pd.DataFrame, out_dir: Path) -> None:
    weighted = build_weighted_sentiment_distribution(df)
    if weighted is None:
        return
    csv_path = out_dir / "sentiment_weighted_distribution.csv"
    png_path = out_dir / "sentiment_weighted_distribution.png"
    weighted.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  💾 {csv_path}")
    plot_weighted_sentiment_distribution(weighted, png_path)
    print(f"  🖼 {png_path}")


def plot_sentiment_distribution(df, save_path):
    """分层样本情感分布饼图。未加权,不代表总体自然口碑。"""
    d = _valid_df(df)
    counts = d["sentiment"].value_counts().reindex(["正面", "中性", "负面"]).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    colors = [SENT_COLORS[s] for s in counts.index]
    total = counts.sum()
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, colors=colors,
        autopct=lambda p: f"{p:.1f}%\n({int(round(p*total/100))}条)",
        startangle=90, textprops={"fontsize": 13},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for t in autotexts:
        t.set_color("white")
        t.set_fontsize(11)
        t.set_fontweight("bold")
    ax.set_title(f"分层样本情感分布(未加权,共 {total} 条)",
                 fontsize=15, pad=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_weighted_sentiment_distribution(weighted: pd.DataFrame, save_path: Path):
    """加权情感分布横向堆叠图。"""
    rows = weighted.copy()
    labels = rows["分组"].astype(str).tolist()
    sents = ["正面", "中性", "负面"]
    mat = np.array([[float(r[f"{s}占比"]) for s in sents]
                    for _, r in rows.iterrows()])

    fig, ax = plt.subplots(figsize=(12, 5.8), dpi=150)
    left = np.zeros(len(rows))
    y = np.arange(len(rows))
    for j, s in enumerate(sents):
        vals = mat[:, j]
        bars = ax.barh(y, vals, left=left, color=SENT_COLORS[s],
                       label=s, edgecolor="white", linewidth=1.2)
        for b, v in zip(bars, vals):
            if v >= 5:
                ax.text(b.get_x() + b.get_width() / 2,
                        b.get_y() + b.get_height() / 2,
                        f"{v:.1f}%", ha="center", va="center",
                        color="white", fontsize=10, fontweight="bold")
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_xlabel("占比 (%)", fontsize=11)
    ax.set_title("按清洗后总体 group×score 分布加权的情感估计",
                 fontsize=15, pad=14)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, fontsize=11, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sentiment_by_group(df, save_path):
    """国内 vs 国际 情感堆叠横向百分比条形图"""
    d = _valid_df(df)
    groups = ["国内游客", "国际游客"]
    sents = ["正面", "中性", "负面"]
    # 百分比矩阵:rows=group, cols=sentiment
    mat = np.zeros((2, 3))
    totals = []
    for i, g in enumerate(groups):
        sub = d[d["group"] == g]
        totals.append(len(sub))
        if len(sub) == 0:
            continue
        for j, s in enumerate(sents):
            mat[i, j] = (sub["sentiment"] == s).sum() / len(sub) * 100

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    left = np.zeros(2)
    y = np.arange(2)
    bars_all = []
    for j, s in enumerate(sents):
        vals = mat[:, j]
        bars = ax.barh(y, vals, left=left, color=SENT_COLORS[s],
                       label=s, edgecolor="white", linewidth=1.2)
        bars_all.append(bars)
        # 百分比标注
        for i, v in enumerate(vals):
            if v > 3:
                ax.text(left[i] + v / 2, i, f"{v:.1f}%",
                        va="center", ha="center",
                        color="white", fontsize=11, fontweight="bold")
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels([f"{g}\n(n={t})" for g, t in zip(groups, totals)],
                       fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_xlabel("占比 (%)", fontsize=11)
    ax.set_title("国内 vs 国际游客 情感分布对比(分层样本)", fontsize=15, pad=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, fontsize=11, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _aspect_sentiment_matrix(df, groups=None):
    """
    展平 aspects 列,返回 DataFrame:aspect × sentiment 的计数矩阵。
    groups=None 表示用全部,否则只看这些组。
    """
    d = _valid_df(df)
    if groups is not None:
        d = d[d["group"].isin(groups)]
    rows = []
    for _, r in d.iterrows():
        for a in _split_aspects(r["aspects"]):
            if a in ASPECTS_ALLOWED:
                rows.append({"aspect": a, "sentiment": r["sentiment"]})
    ex = pd.DataFrame(rows)
    if len(ex) == 0:
        return pd.DataFrame(0, index=ASPECTS_ORDER,
                            columns=["正面", "中性", "负面"])
    mat = pd.crosstab(ex["aspect"], ex["sentiment"])
    mat = mat.reindex(index=ASPECTS_ORDER,
                      columns=["正面", "中性", "负面"]).fillna(0).astype(int)
    return mat


def plot_sentiment_by_aspect(df, save_path):
    """8 方面 × 情感 堆叠柱状图"""
    mat = _aspect_sentiment_matrix(df)

    fig, ax = plt.subplots(figsize=(13, 7), dpi=150)
    x = np.arange(len(ASPECTS_ORDER))
    bottom = np.zeros(len(ASPECTS_ORDER))
    for s in ["正面", "中性", "负面"]:
        vals = mat[s].values
        ax.bar(x, vals, bottom=bottom, color=SENT_COLORS[s],
               label=s, edgecolor="white", linewidth=1.2, width=0.72)
        # 段内标注(只标不太小的段)
        for i, v in enumerate(vals):
            if v >= max(mat.values.max() * 0.04, 5):
                ax.text(i, bottom[i] + v / 2, str(int(v)),
                        ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
        bottom += vals

    # 顶端标注总数
    totals = mat.sum(axis=1).values
    for i, t in enumerate(totals):
        ax.text(i, t + max(totals) * 0.015, f"{int(t)}",
                ha="center", va="bottom", fontsize=10, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(ASPECTS_ORDER, fontsize=11)
    ax.set_ylabel("提及的评论数", fontsize=11)
    ax.set_title("8 个方面的情感分布(堆叠)", fontsize=15, pad=14)
    ax.legend(loc="upper right", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sentiment_by_aspect_grouped(df, save_path):
    """
    报告金图:国内 vs 国际 在 8 方面上的负面比例差异(分组条形图)。
    画"负面占比"而非绝对数,因为两组 n 不一致,比例更公平。
    同时在柱顶标出 n(该方面被该组提及次数)。
    """
    mat_dom = _aspect_sentiment_matrix(df, groups=["国内游客"])
    mat_int = _aspect_sentiment_matrix(df, groups=["国际游客"])

    def neg_ratio(mat):
        total = mat.sum(axis=1).replace(0, np.nan)
        return (mat["负面"] / total * 100).fillna(0).values, total.fillna(0).astype(int).values

    dom_neg, dom_n = neg_ratio(mat_dom)
    int_neg, int_n = neg_ratio(mat_int)

    x = np.arange(len(ASPECTS_ORDER))
    w = 0.38

    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    b1 = ax.bar(x - w / 2, dom_neg, width=w, color=GROUP_COLORS["国内游客"],
                label="国内游客", edgecolor="white", linewidth=1.2)
    b2 = ax.bar(x + w / 2, int_neg, width=w, color=GROUP_COLORS["国际游客"],
                label="国际游客", edgecolor="white", linewidth=1.2)

    # 顶端百分比 + 底部 n
    for i in range(len(ASPECTS_ORDER)):
        ax.text(x[i] - w / 2, dom_neg[i] + 1.2, f"{dom_neg[i]:.0f}%",
                ha="center", fontsize=10, color=GROUP_COLORS["国内游客"])
        ax.text(x[i] + w / 2, int_neg[i] + 1.2, f"{int_neg[i]:.0f}%",
                ha="center", fontsize=10, color=GROUP_COLORS["国际游客"])
        ax.text(x[i] - w / 2, -3.5, f"n={dom_n[i]}",
                ha="center", fontsize=8, color="#666")
        ax.text(x[i] + w / 2, -3.5, f"n={int_n[i]}",
                ha="center", fontsize=8, color="#666")

    ax.set_xticks(x)
    ax.set_xticklabels(ASPECTS_ORDER, fontsize=11)
    ax.set_ylabel("负面评论占比 (%)", fontsize=11)
    ax.set_ylim(-7, max(max(dom_neg), max(int_neg)) * 1.25 + 5)
    ax.set_title("国内 vs 国际游客:8 方面的负面评价占比", fontsize=15, pad=14)
    ax.legend(loc="upper right", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color="#999", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sentiment_trend(df, save_path):
    """按月统计情感占比变化(折线图)"""
    d = _valid_df(df).copy()
    # publishDate 形如 "2026-05-10 15:23:58",取年月
    d["month"] = pd.to_datetime(d["publishDate"], errors="coerce").dt.to_period("M")
    d = d.dropna(subset=["month"])
    if len(d) == 0:
        print("  [WARN] 无可用 publishDate,跳过 trend 图")
        return

    pivot = (d.groupby(["month", "sentiment"]).size()
             .unstack(fill_value=0))
    # 补齐三类
    for s in ["正面", "负面", "中性"]:
        if s not in pivot.columns:
            pivot[s] = 0
    pivot = pivot[["正面", "中性", "负面"]]
    pivot["total"] = pivot.sum(axis=1)
    # 过滤每月至少 5 条,避免小样本噪声
    pivot = pivot[pivot["total"] >= 5]
    if len(pivot) == 0:
        print("  [WARN] 每月样本都少于 5 条,跳过 trend 图")
        return

    pos_pct = pivot["正面"] / pivot["total"] * 100
    neg_pct = pivot["负面"] / pivot["total"] * 100
    months_str = [str(m) for m in pivot.index]

    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    ax.plot(months_str, pos_pct.values, marker="o", color=SENT_COLORS["正面"],
            linewidth=2, label="正面占比")
    ax.plot(months_str, neg_pct.values, marker="s", color=SENT_COLORS["负面"],
            linewidth=2, label="负面占比")

    # 样本量注释
    for i, (m, t) in enumerate(zip(months_str, pivot["total"].values)):
        ax.text(i, -5, f"n={t}", ha="center", fontsize=8, color="#888")

    ax.set_ylim(-10, 100)
    ax.set_ylabel("占比 (%)", fontsize=11)
    ax.set_xlabel("月份", fontsize=11)
    ax.set_title("情感占比按月趋势(每月至少 5 条评论)", fontsize=15, pad=14)
    plt.xticks(rotation=45, ha="right")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_typical_reviews(df, save_path, per_group=3):
    """
    每个 group × sentiment × aspect 组合,抽 per_group 条代表评论。
    挑选规则:优先 intensity='强',再按 content_clean 长度降序(信息更充分)。
    """
    d = _valid_df(df).copy()
    d["_len"] = d["content_clean"].astype(str).map(len)
    d["_inten_rank"] = d["intensity"].map({"强": 0, "中": 1, "弱": 2}).fillna(3)
    # 预解析 aspects,避免在三重循环里反复正则匹配
    d["_aspects_list"] = d["aspects"].map(_split_aspects)

    rows = []
    for g in ["国内游客", "国际游客"]:
        for s in ["正面", "负面", "中性"]:
            base = d[(d["group"] == g) & (d["sentiment"] == s)]
            for a in ASPECTS_ORDER:
                sub = base[base["_aspects_list"].apply(lambda lst: a in lst)]
                if len(sub) == 0:
                    continue
                picks = (sub.sort_values(["_inten_rank", "_len"],
                                         ascending=[True, False])
                            .head(per_group))
                for _, r in picks.iterrows():
                    rows.append({
                        "group": g,
                        "sentiment": s,
                        "aspect": a,
                        "content_clean": r["content_clean"],
                        "key_phrase": r["key_phrase"],
                        "score": r.get("score", ""),
                        "commentId": r["commentId"],
                    })
    out = pd.DataFrame(rows)
    out.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  📝 {save_path}  ({len(out)} 行)")


def write_translation_report(df, save_path):
    """翻译质量报告 md"""
    d = _valid_df(df).copy()

    total = len(d)
    n_tr = int(d["is_translated"].fillna(False).astype(bool).sum())
    n_dom = int((d["group"] == "国内游客").sum())
    n_int = int((d["group"] == "国际游客").sum())
    n_tr_dom = int(((d["group"] == "国内游客") &
                    d["is_translated"].fillna(False).astype(bool)).sum())
    n_tr_int = int(((d["group"] == "国际游客") &
                    d["is_translated"].fillna(False).astype(bool)).sum())

    # 翻译 vs 非翻译 的情感分布(只看国际游客)
    d_int = d[d["group"] == "国际游客"].copy()
    d_int["tr_bool"] = d_int["is_translated"].fillna(False).astype(bool)
    sent_by_tr = pd.crosstab(d_int["tr_bool"], d_int["sentiment"],
                             normalize="index") * 100
    sent_by_tr = sent_by_tr.reindex(columns=["正面", "中性", "负面"]).fillna(0)

    # 抽样:3 条疑似翻译 + 3 条原生中文
    sample_tr = (d_int[d_int["tr_bool"]]
                 .sort_values("content_clean", key=lambda s: s.str.len(),
                              ascending=False)
                 .head(3))
    sample_nat = (d_int[~d_int["tr_bool"]]
                  .sort_values("content_clean", key=lambda s: s.str.len(),
                               ascending=False)
                  .head(3))
    # 如果国际组原生中文样本不足,从国内组取
    if len(sample_nat) < 3:
        extra = (d[d["group"] == "国内游客"]
                 .sort_values("content_clean", key=lambda s: s.str.len(),
                              ascending=False).head(3))
        sample_nat = extra

    lines = []
    lines.append("# 翻译质量报告")
    lines.append("")
    lines.append("本报告基于 Gemini 对每条评论的 `is_translated` 字段判断,"
                 "旨在识别 Trip.com 机器翻译评论对情感分析的潜在影响。")
    lines.append("")
    lines.append("## 一、总体比例")
    lines.append("")
    lines.append(f"- 有效评论总数:**{total}** 条")
    lines.append(f"- 疑似机器翻译:**{n_tr}** 条({n_tr/max(total,1):.1%})")
    lines.append(f"- 国内游客 {n_dom} 条中疑似翻译 {n_tr_dom} 条({n_tr_dom/max(n_dom,1):.1%})")
    lines.append(f"- 国际游客 {n_int} 条中疑似翻译 {n_tr_int} 条({n_tr_int/max(n_int,1):.1%})")
    lines.append("")
    if n_dom > 0 and n_tr_dom / n_dom > 0.05:
        lines.append("> ⚠️ 国内游客组疑似翻译比例偏高,可能是 Gemini 把"
                     "部分生硬表达误判为翻译,也可能是数据分组本身有噪声。")
    else:
        lines.append("> 国内游客组几乎无翻译(符合预期),说明 `is_translated` 的识别"
                     "在国内/国际维度上有很好的区分度。")
    lines.append("")

    lines.append("## 二、翻译 vs 原生 的情感分布(仅国际游客)")
    lines.append("")
    lines.append("|   | 正面 % | 中性 % | 负面 % |")
    lines.append("|---|---|---|---|")
    for key, row in sent_by_tr.iterrows():
        label = "疑似翻译" if bool(key) else "原生中文"
        lines.append(f"| {label} | {row['正面']:.1f} | {row['中性']:.1f} | {row['负面']:.1f} |")
    lines.append("")
    lines.append("翻译评论和原生评论的情感分布若有显著差异,可能暗示:"
                 "(1) 机翻损失了语气信息,导致 Gemini 更倾向判为中性;"
                 "(2) 或者 Trip.com 的国际评论本身就更倾向某种情感。"
                 "具体需要结合下文抽样人工判断。")
    lines.append("")

    lines.append("## 三、抽样对比")
    lines.append("")
    lines.append("### 3.1 疑似机器翻译(3 条)")
    lines.append("")
    for _, r in sample_tr.iterrows():
        lines.append(f"- **id={r['commentId']}** · {r['sentiment']}/{r['intensity']} · "
                     f"key=`{r['key_phrase']}`")
        lines.append(f"  > {str(r['content_clean'])[:300]}")
        lines.append("")
    lines.append("### 3.2 原生中文(3 条)")
    lines.append("")
    for _, r in sample_nat.iterrows():
        lines.append(f"- **id={r['commentId']}** · {r['sentiment']}/{r['intensity']} · "
                     f"key=`{r['key_phrase']}`")
        lines.append(f"  > {str(r['content_clean'])[:300]}")
        lines.append("")

    lines.append("## 四、对情感分析的影响")
    lines.append("")
    lines.append(
        "机器翻译让国际游客的表达趋向以下三点:"
        "(1) **更直译、语义保守** —— 原文里的夸张语气、反讽、俚语容易被磨平;"
        "(2) **指代错误** —— 英语 it / they 在译文里混乱,Gemini 可能误判主语;"
        "(3) **情感中性化倾向** —— 若翻译版的中性比例显著高于原生中文,"
        "说明存在「情感损失」,这部分样本在后续对比研究中应加注释。"
    )
    lines.append("")
    lines.append("建议报告中对「国际游客」的结论做一条脚注:"
                 "**本组样本约 X% 为机器翻译文本,情感判断可能存在一定保守偏差。**")
    lines.append("")

    save_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  📝 {save_path}")


def run_analysis(df, out_dir: Path | None = None):
    """跑完 API 后调用:生成图表、代表评论和加权统计。"""
    set_chinese_font()
    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    valid = _valid_df(df)
    n_err = (df["sentiment"] == "error").sum() if "sentiment" in df.columns else 0
    print(f"\n[analysis] 有效样本 {len(valid)} 条  |  error {n_err} 条")

    if len(valid) == 0:
        print("  [ERROR] 没有有效样本可画图,中止分析")
        return

    print("\n[1/8] sentiment_distribution.png")
    plot_sentiment_distribution(df, out_dir / "sentiment_distribution.png")

    print("[2/8] sentiment_by_group.png")
    plot_sentiment_by_group(df, out_dir / "sentiment_by_group.png")

    print("[3/8] sentiment_by_aspect.png")
    plot_sentiment_by_aspect(df, out_dir / "sentiment_by_aspect.png")

    print("[4/8] sentiment_by_aspect_grouped.png  (报告金图)")
    plot_sentiment_by_aspect_grouped(df, out_dir / "sentiment_by_aspect_grouped.png")

    print("[5/8] sentiment_trend.png")
    plot_sentiment_trend(df, out_dir / "sentiment_trend.png")

    print("[6/8] typical_reviews.csv")
    write_typical_reviews(df, out_dir / "typical_reviews.csv", per_group=3)

    print("[7/8] translation_quality_report.md")
    write_translation_report(df, out_dir / "translation_quality_report.md")

    print("[8/8] sentiment_weighted_distribution.csv/png")
    write_weighted_sentiment_outputs(df, out_dir)


# ============================================================
# 七、主流程
# ============================================================
def main():
    global PROGRESS_FILE  # validate 模式会切换到独立进度文件

    parser = argparse.ArgumentParser(description="阶段3:Gemini 情感与方面分析")
    parser.add_argument("--trial", action="store_true",
                        help="试跑:只处理前10条,打印原始JSON,不落盘,用来调 prompt")
    parser.add_argument("--validate", action="store_true",
                        help="小批:处理前 100 条并落盘,用来抽查质量")
    parser.add_argument("--resume", action="store_true",
                        help="显式续传(默认行为已经是续传,这个参数只是提示)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="跳过API调用,直接从 progress/results 生成图表(阶段3后半段)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --analyze-only: 跳过 API,只基于已有 progress / results 出图表
    if args.analyze_only:
        print("[analyze-only] 跳过 Gemini 调用,直接生成图表与报告")
        df_orig = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
        # 优先用 sentiment_results.csv(已合并),否则用 progress + 原始 data 合并
        results_csv = OUT_DIR / "sentiment_results.csv"
        if results_csv.exists():
            df = pd.read_csv(results_csv, encoding="utf-8-sig")
            print(f"  从 {results_csv} 读入 {len(df)} 条")
            if not validate_comment_ids(df, df_orig, results_csv.name):
                sys.exit(1)
        elif PROGRESS_FILE.exists():
            print(f"  {results_csv.name} 不存在,从 progress + 原始 csv 合并")
            df = finalize_results(df_orig)
            if not validate_comment_ids(df, df_orig, PROGRESS_FILE.name):
                sys.exit(1)
        else:
            print(f"[ERROR] 找不到 {results_csv} 也找不到 {PROGRESS_FILE}。"
                  "请先跑 --validate 或全量", file=sys.stderr)
            sys.exit(1)
        if not validate_sentiment_complete(df, results_csv.name):
            sys.exit(1)
        run_analysis(df)
        _print_output_inventory()
        return

    # ---- 加载数据 ----
    print("[A] 加载数据")
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    print(f"  {len(df)} 条评论")

    # ---- 初始化客户端 ----
    print("[B] 初始化 Gemini 客户端")
    model = init_client()
    print(f"  模型:{MODEL_NAME}  |  batch={BATCH_SIZE}  |  sleep={SLEEP_BETWEEN_BATCHES}s")

    # ---- 试跑模式:只取前10条,打印原始响应 ----
    if args.trial:
        print("\n=== TRIAL 模式:前10条,打印原始 JSON,不落盘 ===")
        df_trial = df.head(10).copy()
        records = call_one_batch(model, df_trial, trial_mode=True)
        print("\n\n📋 解析后(10 条):")
        for r in records:
            # 横向打印一行便于肉眼看
            print(f"  id={r['commentId']:<12} "
                  f"{r['sentiment']}/{r['intensity']:<2} "
                  f"translated={str(r['is_translated']):<5} "
                  f"aspects=[{r['aspects']}]  key='{r['key_phrase']}'")
        print("\n✅ 试跑完成。对照 score、原文肉眼检查质量,满意后去掉 --trial 再跑。")
        return

    # ---- 小批 / 全量共用逻辑 ----
    if args.validate:
        df_work = df.head(100).copy()
        # validate 使用独立进度文件,避免和全量 progress 混用
        PROGRESS_FILE = OUT_DIR / "sentiment_progress_validate.csv"
        validate_results = OUT_DIR / "sentiment_results_validate.csv"
        validate_out_dir = OUT_DIR / "validate"
        print(f"\n=== VALIDATE 模式:前 {len(df_work)} 条 ===")
        print(f"  进度文件: {PROGRESS_FILE.name}(独立于全量)")
        print(f"  验证结果: {validate_results.name}(不会覆盖正式 sentiment_results.csv)")
        print(f"  验证图表目录: {validate_out_dir}")
    else:
        df_work = df.copy()
        validate_results = None
        validate_out_dir = None
        print(f"\n=== FULL 模式:全部 {len(df_work)} 条 ===")

    # ---- 断点续传:剔除已处理的 ----
    current_ids = _comment_id_set(df_work)
    done_ids, _ = load_progress(valid_ids=current_ids)
    work_ids = pd.to_numeric(df_work["commentId"], errors="coerce").astype("Int64")
    df_todo = df_work[~work_ids.isin(done_ids)].copy()
    print(f"  待处理 {len(df_todo)} 条  (已处理 {len(df_work) - len(df_todo)} 条)")

    if len(df_todo) == 0:
        print("✅ 本次目标范围内无新增待处理评论")
    else:
        run_batches(model, df_todo, trial_mode=False)

    # ---- 合并最终结果 ----
    print("\n[Z] 合并最终结果")
    df_final = finalize_results(df_work, out_path=validate_results)

    # ---- 全量或 validate:都顺便出一次图表 ----
    run_analysis(df_final, out_dir=validate_out_dir)

    _print_output_inventory(out_dir=validate_out_dir or OUT_DIR,
                            validate_mode=args.validate)

    print("\n✅ 阶段3 完成。")


def _print_output_inventory(out_dir: Path | None = None, validate_mode: bool = False):
    """打印输出清单,便于人工核对齐全程度。仅列出正式产出。"""
    out_dir = out_dir or OUT_DIR
    # 正式产出白名单(不列临时/日志/中间文件)
    official = {
        "sentiment_distribution.png",
        "sentiment_weighted_distribution.csv",
        "sentiment_weighted_distribution.png",
        "sentiment_by_group.png",
        "sentiment_by_aspect.png",
        "sentiment_by_aspect_grouped.png",
        "sentiment_trend.png",
        "typical_reviews.csv",
        "translation_quality_report.md",
    }
    if not validate_mode:
        official.add("sentiment_results.csv")
    print("\n" + "=" * 60)
    print(f"📦 {'验证' if validate_mode else '正式'}产出清单  ({out_dir}/)")
    missing = []
    for name in sorted(official):
        p = out_dir / name
        if p.exists():
            size_kb = p.stat().st_size / 1024
            print(f"  ✓ {name:<42}  {size_kb:>9.1f} KB")
        else:
            missing.append(name)
            print(f"  ✗ {name:<42}  (缺失)")
    if missing:
        print(f"\n⚠️  有 {len(missing)} 个产出缺失,请检查日志")


if __name__ == "__main__":
    main()
