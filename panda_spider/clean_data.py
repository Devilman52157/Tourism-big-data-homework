# -*- coding: utf-8 -*-
"""
评论数据清洗脚本

输入：data/ctrip_raw.csv（spider_ctrip.py 的产出）
输出：
  - data/panda_reviews_clean.csv  完整字段，UTF-8-sig，Excel 可直接打开
  - data/panda_reviews_for_rost.txt  纯评论文本，每行一条，ANSI(GBK) 编码
                                      （ROST CM6 必须用 ANSI，UTF-8 会乱码）

清洗步骤：
  1. 解析 .NET 风格时间字符串 /Date(1778431717000+0800)/ -> 2026-05-10
  2. 文本预处理：去 emoji / URL / @用户名 / 多余空白
  3. 删除完全重复评论（按 content 去重）
  4. 删除字数 < 10 的评论
  5. 删除疑似灌水：同一段文本出现 >= 3 次、同一用户发布 > 5 条
  6. 输出两个文件 + 在终端打印清洗前后对比
"""
import csv
import glob
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ============== 配置 ==============
# 自动合并 data/ 下所有 ctrip_*.csv（含 ctrip_raw.csv / ctrip_star1.csv 等分层采样产出）
RAW_GLOB = "data/ctrip_*.csv"
CLEAN_CSV_PATH = Path("data/panda_reviews_clean.csv")
ROST_TXT_PATH = Path("data/panda_reviews_for_rost.txt")
OFFTOPIC_CSV_PATH = Path("data/panda_reviews_removed_offtopic.csv")

MIN_LEN = 10               # 评论最少字数
MAX_USER_POSTS = 5         # 同一用户超过这么多条视为水军

# 明显离题词。携程 Trip.com 国际版评论偶尔会混入酒店/机场/商圈点评。
# 只命中这些词不一定删除,还要同时缺少熊猫基地强相关词,以免误删
# "回酒店后下雨"这类仍在谈游览过程的评论。
OFFTOPIC_KEYWORDS = {
    "酒店", "机场", "江北国际机场", "解放碑", "洪崖洞", "入住", "前台",
    "房间", "客房", "早餐", "洗衣房", "礼宾", "商场", "重庆位置",
    "出租车车程", "行李箱", "床", "毛巾",
}

PANDA_SITE_KEYWORDS = {
    "熊猫", "大熊猫", "小熊猫", "红熊猫", "花花", "和花",
    "基地", "园区", "景区", "公园", "观光车", "月亮产房",
    "太阳产房", "西门", "南门", "熊猫谷", "熊猫基地",
}

# ============== 文本清洗 ==============
# emoji 范围（覆盖大部分常用 emoji）
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U00002600-\U000026FF"  # misc symbols
    "\U0001FA70-\U0001FAFF"
    "]+",
    flags=re.UNICODE,
)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
AT_PATTERN = re.compile(r"@[\w一-龥\-]+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(s: str) -> str:
    """对单条评论文本做基本清洗"""
    if not s:
        return ""
    s = EMOJI_PATTERN.sub("", s)
    s = URL_PATTERN.sub("", s)
    s = AT_PATTERN.sub("", s)
    # 把所有连续空白（含换行）压成一个空格
    s = WHITESPACE_PATTERN.sub(" ", s)
    return s.strip()


def off_topic_reason(text: str) -> str:
    """返回离题原因;空字符串表示保留。"""
    if not isinstance(text, str) or not text.strip():
        return "empty"

    hits_bad = [w for w in OFFTOPIC_KEYWORDS if w in text]
    if not hits_bad:
        return ""

    has_site_signal = any(w in text for w in PANDA_SITE_KEYWORDS)
    if has_site_signal:
        return ""

    return "疑似酒店/机场/商圈评论: " + "、".join(sorted(hits_bad)[:5])


def is_relevant_review(text: str) -> bool:
    """是否保留为熊猫基地相关评论。供采样和校验脚本复用。"""
    return off_topic_reason(text) == ""


def parse_dotnet_date(raw: str) -> str:
    """把 /Date(1778431717000+0800)/ 转成 'YYYY-MM-DD HH:MM:SS'。失败返回原值。"""
    if not raw:
        return ""
    m = re.match(r"/Date\((\d+)([+-]\d+)?\)/", raw)
    if not m:
        return raw
    ts_ms = int(m.group(1))
    tz_str = m.group(2) or "+0000"
    sign = 1 if tz_str[0] == "+" else -1
    tz_hours = int(tz_str[1:3])
    tz_minutes = int(tz_str[3:5]) if len(tz_str) >= 5 else 0
    tz = timezone(sign * timedelta(hours=tz_hours, minutes=tz_minutes))
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=tz)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ============== 主流程 ==============
def main():
    # 1. 读所有 data/ctrip_*.csv 并合并（同 commentId 只留一条）
    raw_files = sorted(glob.glob(RAW_GLOB))
    # 排除清洗后的产出文件，避免循环引用
    raw_files = [p for p in raw_files if "clean" not in p and "panda_reviews" not in p]
    if not raw_files:
        print(f"[error] 找不到任何 {RAW_GLOB}，请先跑 spider_ctrip.py")
        sys.exit(1)

    rows_by_id = {}
    per_file_count = []
    for path in raw_files:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            cnt = 0
            for row in csv.DictReader(f):
                cid = row.get("commentId")
                if cid and cid not in rows_by_id:
                    rows_by_id[cid] = row
                cnt += 1
            per_file_count.append((path, cnt))
    rows = list(rows_by_id.values())
    n_raw = len(rows)
    print("[step 1] 合并以下原始 CSV：")
    for p, c in per_file_count:
        print(f"          {p:<35}  {c} 条")
    print(f"          ----- 跨文件去重后共 {n_raw} 条 -----")

    # 2. 文本清洗 + 时间解析
    for r in rows:
        r["content_clean"] = clean_text(r.get("content", ""))
        r["publishDate"] = parse_dotnet_date(r.get("publishTime", ""))

    # 3. 长度过滤
    before = len(rows)
    rows = [r for r in rows if len(r["content_clean"]) >= MIN_LEN]
    print(f"[step 2] 删除字数 < {MIN_LEN} 的：{before - len(rows)} 条 -> 剩 {len(rows)} 条")

    # 4. 景点相关性过滤:剔除明显混入的酒店/机场/商圈评论
    before = len(rows)
    kept = []
    removed_offtopic = []
    for r in rows:
        reason = off_topic_reason(r["content_clean"])
        if reason:
            r["_drop_reason"] = reason
            removed_offtopic.append(r)
        else:
            kept.append(r)
    rows = kept
    print(f"[step 3] 离题过滤：删除 {before - len(rows)} 条 -> 剩 {len(rows)} 条")

    if removed_offtopic:
        OFFTOPIC_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        fieldnames_removed = list(removed_offtopic[0].keys())
        with OFFTOPIC_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames_removed)
            w.writeheader()
            w.writerows(removed_offtopic)
        print(f"          离题样本审计 -> {OFFTOPIC_CSV_PATH}  ({len(removed_offtopic)} 条)")

    # 5. 内容去重（保留第一次出现）
    before = len(rows)
    seen_text = set()
    deduped = []
    for r in rows:
        c = r["content_clean"]
        if c in seen_text:
            continue
        seen_text.add(c)
        deduped.append(r)
    rows = deduped
    print(f"[step 4] 内容去重：删除 {before - len(rows)} 条 -> 剩 {len(rows)} 条")

    # 6. 灌水过滤
    #   同一用户发布 > MAX_USER_POSTS 条
    from collections import Counter
    user_counter = Counter(r.get("userNick", "") for r in rows)
    spam_users = {u for u, c in user_counter.items() if c > MAX_USER_POSTS and u}
    if spam_users:
        before = len(rows)
        rows = [r for r in rows if r.get("userNick") not in spam_users]
        print(f"[step 5] 删除疑似水军用户 ({len(spam_users)} 个)：{before - len(rows)} 条 -> 剩 {len(rows)} 条")
    else:
        print(f"[step 5] 未发现单用户超过 {MAX_USER_POSTS} 条的水军")

    # 7. 输出 panda_reviews_clean.csv（完整字段）
    fieldnames = list(rows[0].keys()) if rows else []
    with CLEAN_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[output] 干净 CSV -> {CLEAN_CSV_PATH}  ({len(rows)} 条)")

    # 8. 输出 panda_reviews_for_rost.txt（GBK，每行一条）
    #    GBK 不支持的字符直接丢弃（errors='ignore'），避免 ROST 读取报错
    with ROST_TXT_PATH.open("w", encoding="gbk", errors="ignore", newline="\r\n") as f:
        for r in rows:
            line = r["content_clean"].replace("\r", " ").replace("\n", " ")
            if line:
                f.write(line + "\r\n")
    print(f"[output] ROST TXT (GBK) -> {ROST_TXT_PATH}  ({len(rows)} 条)")

    # 9. 终端打印数据概览
    if rows:
        lengths = [len(r["content_clean"]) for r in rows]
        avg_len = sum(lengths) / len(lengths)
        dates = [r["publishDate"] for r in rows if r.get("publishDate") and r["publishDate"][0].isdigit()]
        dates.sort()
        print()
        print("=" * 50)
        print(f"清洗前: {n_raw} 条")
        print(f"清洗后: {len(rows)} 条 ({len(rows)/n_raw*100:.1f}%)")
        print(f"平均字数: {avg_len:.1f}  (最长 {max(lengths)}, 最短 {min(lengths)})")
        if dates:
            print(f"评论时间范围: {dates[0]}  ~  {dates[-1]}")
        # 评分分布快览
        score_counter = Counter()
        for r in rows:
            try:
                s = int(float(r.get("score") or 0))
                score_counter[s] += 1
            except ValueError:
                pass
        print("评分分布:")
        for s in sorted(score_counter.keys(), reverse=True):
            print(f"  {s} 星: {score_counter[s]} 条 ({score_counter[s]/len(rows)*100:.1f}%)")
        print("=" * 50)


if __name__ == "__main__":
    main()
