# -*- coding: utf-8 -*-
"""
validate_outputs.py
===================
Lightweight consistency checks for the panda analysis pipeline.

Usage:
    python validate_outputs.py

The script only reads files. It exits with code 1 when a blocking data or
output mismatch is found.
"""

import re
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).parent
SPIDER_DIR = HERE.parent / "panda_spider"
sys.path.insert(0, str(SPIDER_DIR))
from clean_data import off_topic_reason  # noqa: E402

SAMPLE = HERE / "data" / "panda_sample_2000.csv"
CLEAN = SPIDER_DIR / "data" / "panda_reviews_clean.csv"
SENTIMENT = HERE / "output" / "3_sentiment" / "sentiment_results.csv"
WEIGHTED_SENTIMENT = HERE / "output" / "3_sentiment" / "sentiment_weighted_distribution.csv"
DIMENSIONS = HERE / "output" / "4_themes" / "dimensions_table.csv"

REQUIRED_STAGE_OUTPUTS = [
    HERE / "output" / "1_word_freq" / "all_top50.csv",
    HERE / "output" / "1_word_freq" / "domestic_top50.csv",
    HERE / "output" / "1_word_freq" / "intl_top50.csv",
    HERE / "output" / "2_network" / "centrality_all.csv",
    HERE / "output" / "2_network" / "centrality_comparison.csv",
    SENTIMENT,
    WEIGHTED_SENTIMENT,
    HERE / "output" / "3_sentiment" / "typical_reviews.csv",
    DIMENSIONS,
    HERE / "output" / "4_themes" / "full_analysis_report.md",
]

ALLOWED_SENTIMENTS = {"正面", "中性", "负面"}
ALLOWED_INTENSITIES = {"强", "中", "弱"}
EXPECTED_GROUPS = {"国内游客": 1000, "国际游客": 1000}


def _read_csv(path: Path, usecols=None) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", usecols=usecols)


def _id_set(df: pd.DataFrame) -> set[int]:
    return set(pd.to_numeric(df["commentId"], errors="coerce").dropna().astype("int64"))


class Validator:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        if condition:
            print(f"[OK] {message}")
        else:
            self.errors.append(message)
            print(f"[FAIL] {message}")

    def warn_if(self, condition: bool, message: str) -> None:
        if condition:
            self.warnings.append(message)
            print(f"[WARN] {message}")

    def finish(self) -> None:
        print("\n" + "=" * 60)
        print(f"blocking_errors={len(self.errors)}  warnings={len(self.warnings)}")
        if self.errors:
            print("\nBlocking errors:")
            for item in self.errors:
                print(f"- {item}")
            sys.exit(1)
        print("All blocking checks passed.")


def check_required_files(v: Validator) -> None:
    for path in [SAMPLE, CLEAN, *REQUIRED_STAGE_OUTPUTS]:
        v.check(path.exists(), f"required file exists: {path.relative_to(HERE.parent)}")


def check_sample(v: Validator) -> pd.DataFrame | None:
    if not SAMPLE.exists():
        return None
    sample = _read_csv(SAMPLE)
    v.check(len(sample) == 2000, "sample has exactly 2000 rows")
    v.check(sample["commentId"].nunique() == len(sample), "sample commentId has no duplicates")
    v.check(not sample["content_clean"].fillna("").astype(str).str.strip().eq("").any(),
            "sample content_clean has no empty text")
    v.check(sample["score"].isin([1, 2, 3, 4, 5]).all(), "sample score is within 1-5")

    group_counts = sample["group"].value_counts().to_dict()
    v.check(group_counts == EXPECTED_GROUPS,
            f"sample group counts match {EXPECTED_GROUPS}: {group_counts}")

    dates = pd.to_datetime(sample["publishDate"], errors="coerce")
    v.check(dates.notna().all(), "sample publishDate is parseable")
    v.warn_if((dates > pd.Timestamp.now() + pd.Timedelta(days=1)).any(),
              "sample contains future publishDate values")

    reasons = sample["content_clean"].fillna("").astype(str).map(off_topic_reason)
    v.check(reasons.eq("").all(), "sample contains no obvious hotel/airport off-topic reviews")
    return sample


def check_clean_relation(v: Validator, sample: pd.DataFrame | None) -> None:
    if sample is None or not CLEAN.exists():
        return
    clean = _read_csv(CLEAN)
    v.check(clean["commentId"].nunique() == len(clean), "clean commentId has no duplicates")
    reasons = clean["content_clean"].fillna("").astype(str).map(off_topic_reason)
    v.check(reasons.eq("").all(), "clean data contains no obvious hotel/airport off-topic reviews")
    v.check(_id_set(sample).issubset(_id_set(clean)), "sample commentIds are all present in clean data")

    cols = ["commentId", "score", "content_clean", "publishDate", "fromTypeText", "ipLocatedName"]
    merged = sample[cols].merge(clean[cols], on="commentId", how="left", suffixes=("_sample", "_clean"))
    for col in cols[1:]:
        left = merged[f"{col}_sample"].fillna("<NA>").astype(str)
        right = merged[f"{col}_clean"].fillna("<NA>").astype(str)
        v.check((left == right).all(), f"sample {col} matches clean data")


def check_raw_relation(v: Validator) -> None:
    """确认当前 clean 文件可由当前 raw 星级文件追溯。"""
    if not CLEAN.exists():
        return
    data_dir = SPIDER_DIR / "data"
    raw_files = sorted(
        p for p in data_dir.glob("ctrip_*.csv")
        if "clean" not in p.name and "panda_reviews" not in p.name
    )
    v.check(bool(raw_files), "raw ctrip_*.csv files exist for reproducibility")
    if not raw_files:
        return

    raw_parts = [_read_csv(p, usecols=["commentId"]) for p in raw_files]
    raw_ids = set()
    for part in raw_parts:
        raw_ids |= _id_set(part)
    clean_ids = _id_set(_read_csv(CLEAN, usecols=["commentId"]))
    missing = clean_ids - raw_ids
    v.check(not missing,
            f"clean commentIds are traceable to current raw CSV files "
            f"(missing_from_raw={len(missing)})")


def check_sentiment(v: Validator, sample: pd.DataFrame | None) -> pd.DataFrame | None:
    if sample is None or not SENTIMENT.exists():
        return None
    sent = _read_csv(SENTIMENT)
    v.check(len(sent) == len(sample), "sentiment_results row count matches sample")
    v.check(_id_set(sent) == _id_set(sample), "sentiment_results commentIds match sample exactly")
    v.check(sent["commentId"].nunique() == len(sent), "sentiment_results commentId has no duplicates")
    v.check(sent["sentiment"].isin(ALLOWED_SENTIMENTS).all(), "sentiment values are valid")
    v.check(sent["intensity"].isin(ALLOWED_INTENSITIES).all(), "intensity values are valid")
    v.check(sent["sentiment"].notna().all(), "sentiment has no missing values")
    if "is_translated" in sent.columns:
        intl = sent["group"] == "国际游客"
        tr = sent["is_translated"].fillna(False).astype(bool)
        pct_intl = (tr & intl).sum() / max(intl.sum(), 1) * 100
        print(f"[INFO] international translated ratio: {pct_intl:.1f}%")
    return sent


def check_weighted_sentiment(v: Validator) -> None:
    if not WEIGHTED_SENTIMENT.exists():
        return
    weighted = _read_csv(WEIGHTED_SENTIMENT)
    required = {
        "分组", "估计总体量", "正面估计量", "正面占比",
        "中性估计量", "中性占比", "负面估计量", "负面占比", "样本量",
    }
    v.check(required.issubset(weighted.columns),
            "weighted sentiment table has required columns")
    v.check(len(weighted) == 3,
            "weighted sentiment table contains overall/domestic/international rows")
    pct_cols = ["正面占比", "中性占比", "负面占比"]
    pct_sum = weighted[pct_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    v.check(((pct_sum - 100).abs() <= 0.2).all(),
            "weighted sentiment percentages sum to 100%")
    v.check(pd.to_numeric(weighted["估计总体量"], errors="coerce").gt(0).all(),
            "weighted sentiment population estimates are positive")


def check_dimensions(v: Validator) -> None:
    if not DIMENSIONS.exists():
        return
    dims = _read_csv(DIMENSIONS)
    required = {"维度名称", "包含的高频词", "覆盖评论数", "整体情感得分",
                "国内情感得分", "国际情感得分"}
    v.check(required.issubset(dims.columns), "dimensions_table has required columns")
    v.check(len(dims) == 5, "dimensions_table contains exactly 5 dimensions")
    score_cols = ["整体情感得分", "国内情感得分", "国际情感得分"]
    for col in score_cols:
        numeric = pd.to_numeric(dims[col], errors="coerce")
        v.check(numeric.between(-1, 1).all(), f"{col} is within [-1, 1]")


def check_stale_hardcoded_values(v: Validator) -> None:
    stale_patterns = {
        "64.7%": "old machine-translation ratio",
        "2170": "old 熊猫 frequency",
        "777 个停用词": "old stopword count",
    }
    for script_name in ["make_report.py", "make_ppt.py"]:
        path = HERE / script_name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for pattern, label in stale_patterns.items():
            v.warn_if(re.search(re.escape(pattern), text) is not None,
                      f"{script_name} still contains {label}: {pattern}")


def main() -> None:
    v = Validator()
    check_required_files(v)
    sample = check_sample(v)
    check_raw_relation(v)
    check_clean_relation(v, sample)
    check_sentiment(v, sample)
    check_weighted_sentiment(v)
    check_dimensions(v)
    check_stale_hardcoded_values(v)
    v.finish()


if __name__ == "__main__":
    main()
