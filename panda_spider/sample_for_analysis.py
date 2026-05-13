# -*- coding: utf-8 -*-
"""
sample_for_analysis.py
======================
把爬虫清洗后的 panda_reviews_clean.csv 转成分析阶段直接可用的
panda_sample_2000.csv(国内 1000 条 + 国际 1000 条)。

这是爬虫项目(panda_spider)与分析项目(panda_analysis)之间的
"最后一公里"——此前该步骤靠手工完成,缺失会导致整个工作流无法端到端复现。

分组规则(写死在此,避免歧义):
    国际游客 =
        fromTypeText == "来自Trip.com"
        OR ipLocatedName ∈ FOREIGN_COUNTRIES
    国内游客 = 其他一切(含港澳台、未知、NaN)

    * 港澳台评论文本为中文原生、无机翻痕迹,并走"来自订单"接口,
      视作与大陆文化背景同源,归入国内组。

抽样策略:
    - 各组按 score(1-5 星)分层采样,每星抽配额见 QUOTA_DOM / QUOTA_INT。
    - 配额不足时就收全量,超过时按 seed 做随机抽样。
    - 默认先过滤 publishDate >= 2024-01-01,保证样本集中在近年时段。
    - 该样本用于保证各评分层都有足够评论可比较,不是自然总体比例。
      若要描述整体口碑,应使用分析阶段输出的加权情感分布。

输入: panda_spider/data/panda_reviews_clean.csv(由 clean_data.py 产出)
输出: panda_analysis/data/panda_sample_2000.csv(UTF-8-sig)

用法:
    python sample_for_analysis.py
    python sample_for_analysis.py --seed 42
    python sample_for_analysis.py --dry-run   # 只打印统计,不落盘
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from clean_data import off_topic_reason

sys.stdout.reconfigure(encoding="utf-8")

# ==================== 路径 ====================
# 默认从 panda_spider/data 读,写到隔壁 panda_analysis/data
SCRIPT_DIR = Path(__file__).parent
DEFAULT_SRC = SCRIPT_DIR / "data" / "panda_reviews_clean.csv"
DEFAULT_DST = SCRIPT_DIR.parent / "panda_analysis" / "data" / "panda_sample_2000.csv"

# ==================== 分组规则 ====================
# 国外 IP 白名单:来自 panda_reviews_clean.csv 的 ipLocatedName 字段中,
# 明确属于境外的取值。港澳台、"未知"、NaN 归入国内组。
# 新增国家时在此追加即可。
FOREIGN_COUNTRIES = {
    "新加坡", "日本", "马来西亚", "泰国", "美国", "澳大利亚",
    "英国", "德国", "荷兰", "韩国", "瑞士", "奥地利",
    "尼泊尔", "斯里兰卡", "加拿大",
}


def tag_group(row) -> str:
    """给一行评论打 group 标签。"""
    ft = row.get("fromTypeText")
    ip = row.get("ipLocatedName")
    # Trip.com 来源 = 国际
    if ft == "来自Trip.com":
        return "国际游客"
    # IP 在境外白名单 = 国际
    if isinstance(ip, str) and ip in FOREIGN_COUNTRIES:
        return "国际游客"
    # 其余一切(国内省份、港澳台、未知、NaN)= 国内
    return "国内游客"


# ==================== 分层配额 ====================
# 国内组:为了保留足够差评/中评案例,低分层有意提高可见度。
# 国际组:1-2 星样本极少(各 ~40 条),高分样本占大头,直接按可用量上限抽。
# 注意:该配额服务于组间比较和低分文本分析,整体情感比例需在
# panda_analysis/03_sentiment.py 中按清洗后总体 group×score 分布加权解释。
QUOTA_DOM = {1: 300, 2: 120, 3: 290, 4: 180, 5: 110}   # 合计 1000
QUOTA_INT = {1: 30,  2: 20,  3: 150, 4: 330, 5: 470}   # 合计 1000


def stratified_sample(df: pd.DataFrame, quota: dict, seed: int) -> pd.DataFrame:
    """对一个 group 的 df 按 score 做分层采样。不足就全取,超过就随机取。"""
    parts = []
    for score, n in quota.items():
        sub = df[df["score"] == score]
        if len(sub) == 0:
            print(f"    [warn] score={score} 无样本,跳过")
            continue
        if len(sub) <= n:
            parts.append(sub)
            print(f"    score={score}: 需要 {n},实际只有 {len(sub)} 条,全取")
        else:
            parts.append(sub.sample(n=n, random_state=seed))
            print(f"    score={score}: {len(sub)} 条池中抽 {n} 条")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def assert_no_offtopic(df: pd.DataFrame, label: str) -> None:
    """采样前后都做硬性检查,避免离题评论进入分析。"""
    if "content_clean" not in df.columns:
        print(f"[ERROR] {label} 缺少 content_clean 列,无法检查离题评论",
              file=sys.stderr)
        sys.exit(1)

    reasons = df["content_clean"].fillna("").astype(str).map(off_topic_reason)
    bad = df[reasons != ""].copy()
    if bad.empty:
        print(f"      {label}: 未发现明显离题评论")
        return

    bad["_offtopic_reason"] = reasons[reasons != ""].values
    print(f"[ERROR] {label} 发现 {len(bad)} 条明显离题评论,请先重跑 clean_data.py 或人工处理。",
          file=sys.stderr)
    cols = [c for c in ["commentId", "score", "fromTypeText",
                        "ipLocatedName", "content_clean",
                        "_offtopic_reason"] if c in bad.columns]
    for line in bad[cols].head(10).to_string(index=False, max_colwidth=80).splitlines():
        print("        " + line, file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC),
                    help="清洗后的 CSV 路径(默认 panda_spider/data/panda_reviews_clean.csv)")
    ap.add_argument("--dst", default=str(DEFAULT_DST),
                    help="输出样本路径(默认 panda_analysis/data/panda_sample_2000.csv)")
    ap.add_argument("--seed", type=int, default=42,
                    help="抽样随机种子,默认 42。换种子即可得到不同样本。")
    ap.add_argument("--min-date", default="2024-01-01",
                    help="只保留 publishDate >= 该日期的评论,默认 2024-01-01;"
                         "传 '' 或 'none' 表示不过滤")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印统计,不落盘(用于检视分组/配额)")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.exists():
        print(f"[ERROR] 源文件不存在: {src}\n"
              f"请先在 panda_spider/ 下跑 clean_data.py 产出 panda_reviews_clean.csv",
              file=sys.stderr)
        sys.exit(1)

    # 1. 读数据
    print(f"[1/4] 读入 {src}")
    df = pd.read_csv(src, encoding="utf-8-sig")
    print(f"      共 {len(df)} 条评论")
    assert_no_offtopic(df, "清洗表")

    # 时间过滤:默认只保留 2024-01-01 之后,与原分析时段保持一致
    if args.min_date and args.min_date.lower() != "none":
        before = len(df)
        df = df[df["publishDate"] >= args.min_date].copy()
        print(f"      时间过滤 publishDate >= {args.min_date}: "
              f"{before} → {len(df)} 条")

    # 2. 打 group 标签
    print("[2/4] 打 group 标签(规则:Trip.com 或 国外 IP → 国际)")
    df["group"] = df.apply(tag_group, axis=1)
    vc = df["group"].value_counts().to_dict()
    print(f"      国内池: {vc.get('国内游客', 0)} 条")
    print(f"      国际池: {vc.get('国际游客', 0)} 条")

    # 3. 分层抽样
    print(f"[3/4] 分层抽样(seed={args.seed})")
    df_dom_pool = df[df["group"] == "国内游客"].copy()
    df_int_pool = df[df["group"] == "国际游客"].copy()

    print("  国内组:")
    dom = stratified_sample(df_dom_pool, QUOTA_DOM, args.seed)
    print("  国际组:")
    intl = stratified_sample(df_int_pool, QUOTA_INT, args.seed)

    sample = pd.concat([dom, intl], ignore_index=True)
    # 按时间排序,让阶段 3 的趋势图更自然
    sample = sample.sort_values("publishDate").reset_index(drop=True)
    assert_no_offtopic(sample, "最终样本")

    # 4. 概览 + 落盘
    print("[4/4] 最终样本概览")
    print(f"      总数: {len(sample)}")
    print("      按 group × score 交叉分布:")
    cross = pd.crosstab(sample["group"], sample["score"], margins=True)
    for line in cross.to_string().splitlines():
        print("        " + line)
    print(f"      时间范围: {sample['publishDate'].min()}  ~  "
          f"{sample['publishDate'].max()}")
    print(f"      国内 fromTypeText 分布: "
          f"{sample[sample['group']=='国内游客']['fromTypeText'].value_counts(dropna=False).to_dict()}")
    print(f"      国际 fromTypeText 分布: "
          f"{sample[sample['group']=='国际游客']['fromTypeText'].value_counts(dropna=False).to_dict()}")

    if args.dry_run:
        print("\n[dry-run] 仅统计,不落盘")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(dst, index=False, encoding="utf-8-sig")
    print(f"\n✅ 已写入 {dst}")
    print(f"   下一步:切到 panda_analysis/ 跑 python 01_word_freq.py 开始分析")


if __name__ == "__main__":
    main()
