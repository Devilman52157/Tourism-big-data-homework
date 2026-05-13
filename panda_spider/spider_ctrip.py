# -*- coding: utf-8 -*-
"""
携程评论爬虫 —— 成都大熊猫繁育研究基地

接口：POST https://m.ctrip.com/restapi/soa2/13444/json/getCommentCollapseList
关键参数：
    arg.poiId = 76342         # 大熊猫基地 POI（从浏览器 F12 抓包得到）
    arg.pageIndex             # 翻页（1 起步）
    arg.pageSize              # 每页条数（10/50 都试过可用）
    arg.sortType = 3          # 3 = 按时间倒序
    arg.starType              # 0=全部, 1=1星, 2=2星, 3=3星, 4=4星, 5=5星
    arg.commentTagId = -11    # 全部标签
    arg.channelType = 2       # PC 渠道
    arg.collapseType = 0      # 不折叠
    arg.sourceType = 1        # 来源 = 真实游客

输出：data/ctrip_raw.csv，UTF-8-sig，可 Excel 直接打开。
设计：
  - 边爬边写，每页 flush，断点续爬（commentId 去重）
  - 随机 sleep 2~4 秒
  - 重试 3 次，失败记录到 data/ctrip_failed_pages.txt
  - 触发反爬（403/429/空响应连续 3 次）就主动停止
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import requests

# Windows PowerShell 终端默认 GBK，强制 stdout 用 utf-8 防止打印中文崩
sys.stdout.reconfigure(encoding="utf-8")

# ===================== 配置 =====================
POI_ID = 76342
ENDPOINT = "https://m.ctrip.com/restapi/soa2/13444/json/getCommentCollapseList"

# 注：以下 cookie 来自浏览器抓包，已剔除登录态相关字段（login_uid/cticket/AHeadUserInfo 等）
# 仅保留访客指纹类字段（UBT_VID/GUID/_bfa 等），用于让接口认为是正常浏览器请求
COOKIE = (
    "UBT_VID=1778429415496.f6cbd3Zr6gHz; "
    "GUID=09031089313121820049; "
    "_RGUID=badf0d85-b80f-4912-b1d2-301c19dde5d7; "
    "MKT_Pagesource=PC; "
    "_bfa=1.1778429415496.f6cbd3Zr6gHz.1.1778440641298.1778440672825.2.7.290510"
)

HEADERS = {
    # 抓包是手机 UA，沿用即可（接口对 UA 不敏感）
    "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Mobile Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Content-Type": "application/json",
    "Origin": "https://you.ctrip.com",
    "Referer": "https://you.ctrip.com/",
    "Cookie": COOKIE,
    "cookieorigin": "https://you.ctrip.com",
}

# 输出 CSV 表头
CSV_FIELDS = [
    "commentId",       # 评论唯一 ID（用于去重和断点续爬）
    "page",            # 来自第几页（方便回查）
    "content",         # 评论文本
    "score",           # 评分 1~5
    "publishTime",     # 发布时间
    "publishTypeTag",  # "2026-05-11 发布点评" 这种字符串
    "userNick",        # 昵称
    "ipLocatedName",   # IP 属地
    "touristTypeDisplay",  # 出游类型（家庭/情侣...）
    "usefulCount",     # 有用数
    "imageCount",      # 图片数（不存图，只记数量）
    "videoCount",      # 视频数
    "fromTypeText",    # 来源文案
    # 追溯字段:接口偶尔会返回跨产品/跨资源评论,这些字段用于回查。
    # 已有旧 CSV 断点续爬时会自动沿用旧表头,新字段只写入新建 CSV。
    "resourceId",
    "poiId",
    "businessId",
    "sourceName",
    "rawItemJson",
]


# ===================== 网络层 =====================
def build_payload(page_index: int, page_size: int, star_type: int = 0) -> dict:
    """构造一次请求的 JSON 体。head 里的 cid 来自抓包，固定就行。
    star_type: 0=全部, 1=1星, 2=2星, 3=3星, 4=4星, 5=5星。"""
    return {
        "arg": {
            "channelType": 2,
            "collapseType": 0,
            "commentTagId": -11,
            "pageIndex": page_index,
            "pageSize": page_size,
            "poiId": POI_ID,
            "sourceType": 1,
            "sortType": 3,
            "starType": star_type,
        },
        "head": {
            "cid": "09031089313121820049",
            "ctok": "", "cver": "1.0", "lang": "01",
            "sid": "8888", "syscode": "09",
            "auth": "", "xsid": "", "extension": [],
        },
    }


def fetch_page(page_index: int, page_size: int, star_type: int = 0, retries: int = 3) -> dict | None:
    """请求单页评论。失败重试，全失败返回 None。"""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(
                ENDPOINT, headers=HEADERS,
                json=build_payload(page_index, page_size, star_type),
                timeout=20,
            )
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}"
                # 403/429 = 反爬触发
                if r.status_code in (403, 429):
                    print(f"  [page {page_index}] {r.status_code} 反爬触发，停止重试")
                    return None
            else:
                data = r.json()
                # 业务层错误码也要看
                if data.get("code") not in (200, 0, None):
                    last_err = f"biz code={data.get('code')} msg={data.get('msg')}"
                else:
                    return data
        except requests.RequestException as e:
            last_err = repr(e)
        except json.JSONDecodeError as e:
            last_err = f"json decode: {e}"
        # 重试前等待
        time.sleep(1.5 * attempt)
    print(f"  [page {page_index}] 全部 {retries} 次重试失败: {last_err}")
    return None


def parse_items(data: dict, page_index: int, keep_raw_item: bool = False) -> list[dict]:
    """从 API 返回里抽出我们要落库的字段。"""
    rows = []
    items = ((data or {}).get("result") or {}).get("items") or []
    for it in items:
        ui = it.get("userInfo") or {}
        rows.append({
            "commentId": it.get("commentId"),
            "page": page_index,
            "content": (it.get("content") or "").strip(),
            "score": it.get("score"),
            "publishTime": it.get("publishTime"),
            "publishTypeTag": it.get("publishTypeTag"),
            "userNick": ui.get("userNick"),
            "ipLocatedName": it.get("ipLocatedName"),
            "touristTypeDisplay": it.get("touristTypeDisplay"),
            "usefulCount": it.get("usefulCount"),
            "imageCount": len(it.get("images") or []),
            "videoCount": len(it.get("videos") or []),
            "fromTypeText": it.get("fromTypeText"),
            "resourceId": it.get("resourceId") or it.get("businessResourceId"),
            "poiId": it.get("poiId") or it.get("businessId"),
            "businessId": it.get("businessId"),
            "sourceName": it.get("sourceName") or it.get("resourceName") or it.get("poiName"),
            "rawItemJson": (
                json.dumps(it, ensure_ascii=False, separators=(",", ":"))
                if keep_raw_item else ""
            ),
        })
    return rows


# ===================== CSV 层 =====================
def load_existing_ids(csv_path: Path) -> set:
    """读取已有 CSV，返回已抓取的 commentId 集合，用于断点续爬去重。"""
    if not csv_path.exists():
        return set()
    seen = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("commentId")
            if cid:
                seen.add(cid)
    return seen


def open_writer(csv_path: Path):
    """打开 CSV 追加写。新文件先写表头。返回 (file, writer)。"""
    is_new = not csv_path.exists()
    fieldnames = CSV_FIELDS
    if not is_new:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as existing:
            reader = csv.reader(existing)
            header = next(reader, None)
            if header:
                fieldnames = header
    f = csv_path.open("a", encoding="utf-8-sig", newline="")
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    if is_new:
        w.writeheader()
    return f, w


def append_raw_jsonl(path: Path, page: int, star: int, data: dict) -> None:
    """可选保存原始 item JSON,便于追查离题评论是否来自接口本身。"""
    items = ((data or {}).get("result") or {}).get("items") or []
    if not items:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for it in items:
            record = {
                "endpointPoiId": POI_ID,
                "page": page,
                "star": star,
                "commentId": it.get("commentId"),
                "item": it,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ===================== 主流程 =====================
def main():
    ap = argparse.ArgumentParser(description="Ctrip panda base review spider")
    ap.add_argument("--start", type=int, default=1, help="起始页码")
    ap.add_argument("--max-pages", type=int, default=200,
                    help="最多翻多少页（默认 200，pageSize=50 时上限 ~10000 条）")
    ap.add_argument("--page-size", type=int, default=50, help="每页条数（10/50）")
    ap.add_argument("--star", type=int, default=0,
                    help="按星筛选：0=全部, 1=1星, 2=2星, 3=3星, 4=4星, 5=5星")
    ap.add_argument("--sleep-min", type=float, default=2.0)
    ap.add_argument("--sleep-max", type=float, default=4.0)
    ap.add_argument("--out", type=str, default="data/ctrip_raw.csv")
    ap.add_argument("--keep-raw-item", action="store_true",
                    help="在 CSV 里额外保存每条 item 的原始 JSON(文件会明显变大)")
    ap.add_argument("--raw-jsonl", type=str, default="",
                    help="把接口返回的原始 items 追加保存为 JSONL,便于追溯离题数据")
    args = ap.parse_args()

    csv_path = Path(args.out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path = csv_path.parent / "ctrip_failed_pages.txt"

    seen_ids = load_existing_ids(csv_path)
    print(f"[init] 已有 commentId 数: {len(seen_ids)}")

    f, writer = open_writer(csv_path)

    total_new = 0
    consecutive_empty = 0
    consecutive_fail = 0
    declared_total = None
    t0 = time.time()

    try:
        print(f"[init] starType={args.star}  pageSize={args.page_size}  max_pages={args.max_pages}")
        for page in range(args.start, args.start + args.max_pages):
            data = fetch_page(page, args.page_size, args.star)
            if data is None:
                consecutive_fail += 1
                with failed_path.open("a", encoding="utf-8") as fp:
                    fp.write(f"{page}\n")
                if consecutive_fail >= 3:
                    print("[stop] 连续 3 页失败，主动停止")
                    break
                continue
            consecutive_fail = 0

            # 第一次拿到响应时记录 totalCount，便于估算
            if declared_total is None:
                declared_total = ((data.get("result") or {}).get("totalCount"))
                print(f"[init] API 声称 totalCount = {declared_total}")

            if args.raw_jsonl:
                append_raw_jsonl(Path(args.raw_jsonl), page, args.star, data)

            rows = parse_items(data, page, keep_raw_item=args.keep_raw_item)
            if not rows:
                consecutive_empty += 1
                print(f"[page {page}] 空，连续空页数={consecutive_empty}")
                if consecutive_empty >= 3:
                    print("[stop] 连续 3 页空，应已到达分页深度上限，停止")
                    break
                # 空页不需要 sleep 太久
                time.sleep(random.uniform(1.0, 2.0))
                continue
            consecutive_empty = 0

            page_new = 0
            for row in rows:
                cid = str(row["commentId"])
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                writer.writerow(row)
                page_new += 1
            f.flush()
            total_new += page_new
            print(f"[page {page:>4}] got {len(rows):>2} items, new {page_new:>2}, total_new={total_new}")

            # 礼貌延时
            time.sleep(random.uniform(args.sleep_min, args.sleep_max))
    except KeyboardInterrupt:
        print("\n[stop] 用户中断")
    finally:
        f.close()

    elapsed = time.time() - t0
    print(f"\n[done] 本次新增 {total_new} 条，CSV 累计 {len(seen_ids)} 条，"
          f"耗时 {elapsed:.1f}s，输出 -> {csv_path}")


if __name__ == "__main__":
    main()
