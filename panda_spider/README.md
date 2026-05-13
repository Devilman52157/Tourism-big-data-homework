# 成都大熊猫繁育研究基地 评论数据采集

课程作业：从携程采集游客评论文本，用于词频分析、社会语义网络分析、情感分析。

## 数据源说明

- **携程**：评论接口 `m.ctrip.com/restapi/soa2/13444/json/getCommentCollapseList`
  - 大熊猫基地 `poiId = 76342`（从 PC 详情页 F12 抓包得到）
  - 总池约 6.9 万条，本项目目标抓取约 1 万条
- **马蜂窝**：原计划同步采集 500 条，但全站受 Akamai Bot Manager 保护
  （所有 HTML/接口请求均返回 HTTP 202 + probe.js 指纹挑战），
  requests 直连无法穿过，已放弃马蜂窝部分。

## 项目结构

```
panda_spider/
├── spider_ctrip.py         # 携程评论爬虫
├── clean_data.py           # 数据清洗（输出 CSV + ROST TXT）
├── stats.py                # 描述统计 + 可视化（可选）
├── sample_for_analysis.py  # 采样给分析项目（产出 panda_sample_2000.csv）
├── probe_final.py          # 阶段 1 接口探针（保留作为接口发现过程的证据）
├── requirements.txt
├── data/
│   ├── ctrip_star{1..5}.csv        # 爬虫按星级分层采样产出
│   ├── panda_reviews_clean.csv              # 清洗后完整字段（UTF-8-sig）
│   ├── panda_reviews_removed_offtopic.csv   # 被剔除的疑似酒店/机场/商圈离题评论
│   └── panda_reviews_for_rost.txt           # 纯文本，ANSI(GBK)，给 ROST CM6 用
└── figs/                            # stats.py 跑完生成（score/length/monthly 三张图）
```

## 环境准备

需要 Python 3.10+。在项目目录下：

```powershell
# 创建虚拟环境（已创建则跳过）
python -m venv .venv

# 激活
.venv\Scripts\Activate.ps1

# 装依赖
pip install -r requirements.txt
```

## 运行顺序

### 1. 爬取数据 — `spider_ctrip.py`

携程评分严重正向偏（5星占 75%、1-2星仅 5%），按时间排序会全是好评。**推荐分层采样：差评/中评全收 + 好评精选**。

```powershell
# 1星（约 2700 条）
python spider_ctrip.py --star 1 --max-pages 60 --out data/ctrip_star1.csv

# 2星（约 1200 条）
python spider_ctrip.py --star 2 --max-pages 30 --out data/ctrip_star2.csv

# 3星（约 4800 条）
python spider_ctrip.py --star 3 --max-pages 100 --out data/ctrip_star3.csv

# 4星抓 2000 条
python spider_ctrip.py --star 4 --max-pages 40 --out data/ctrip_star4.csv

# 5星抓 2000 条
python spider_ctrip.py --star 5 --max-pages 40 --out data/ctrip_star5.csv
```

或者均衡采样（每星各 1000 条 → 共 5000 条）：
```powershell
python spider_ctrip.py --star 1 --max-pages 20 --out data/ctrip_star1.csv
python spider_ctrip.py --star 2 --max-pages 20 --out data/ctrip_star2.csv
python spider_ctrip.py --star 3 --max-pages 20 --out data/ctrip_star3.csv
python spider_ctrip.py --star 4 --max-pages 20 --out data/ctrip_star4.csv
python spider_ctrip.py --star 5 --max-pages 20 --out data/ctrip_star5.csv
```

主要参数：
| 参数 | 默认 | 说明 |
|---|---|---|
| `--star` | 0 | 评分筛选：**0=全部, 1/2/3/4/5=对应星级** |
| `--start` | 1 | 起始页码（断点续爬时改这里）|
| `--max-pages` | 200 | 最多翻多少页 |
| `--page-size` | 50 | 每页条数 |
| `--sleep-min/--sleep-max` | 2.0 / 4.0 | 每页间随机延时（秒）|
| `--out` | `data/ctrip_star{N}.csv` | 输出路径（按 --star 命名） |

特性：
- **断点续爬**：重新启动时自动读已有 CSV，跳过已抓 `commentId`，从 `--start` 继续
- **失败重试**：网络错误重试 3 次；连续 3 页失败自动停止
- **空页停止**：连续 3 页返回空（到达分页深度上限）自动停止
- **反爬保护**：遇到 403/429 立刻停止，不绕反爬

### 2. 清洗数据 — `clean_data.py`

```powershell
python clean_data.py
```

清洗规则：
1. 解析 `/Date(1778431717000+0800)/` 时间戳为 `YYYY-MM-DD HH:MM:SS`
2. 文本预处理：去 emoji / URL / @用户名 / 多余空白
3. 删除字数 < 10 的评论
4. 删除明显离题评论：命中酒店/机场/商圈词，且不含熊猫基地强相关词的记录
5. 完全重复评论去重（按清洗后文本）
6. 删除疑似水军（同一用户发布 > 5 条）
7. 输出：
   - `data/panda_reviews_clean.csv`（UTF-8-sig，完整字段）
   - `data/panda_reviews_removed_offtopic.csv`（离题评论审计表）
   - `data/panda_reviews_for_rost.txt`（**ANSI/GBK**，每行一条，供 ROST CM6 直接使用）
8. 终端打印清洗前后对比、字数、时间范围、评分分布

### 3. 描述统计 — `stats.py`

```powershell
python stats.py
```

输出：
- 终端打印：总数、来源分布、评分分布、字数、时间范围
- `figs/score_distribution.png` 评分分布柱状图
- `figs/length_distribution.png` 字数分布直方图
- `figs/monthly_trend.png` 月度评论量趋势

### 4. 采样给分析项目 — `sample_for_analysis.py`

把清洗好的 7000+ 条评论转成分析项目直接可用的 `panda_sample_2000.csv`（国内 1000 + 国际 1000）。**本步骤是从 `panda_spider` 通往 `panda_analysis` 的必经环节**，不跑这步 analysis 端拿不到带 `group` 字段的样本。
注意:这 2000 条是分析型分层样本,用于保证低分、中分、高分评论都有足够文本可比较,不能直接把未加权整体情感比例解释为平台总体口碑。分析端会按清洗后总体的 `group × score` 分布生成加权情感统计。
脚本会再次检查 `content_clean`，如果发现明显酒店/机场/商圈离题评论，会直接中止，避免脏数据进入后续分析。

```powershell
python sample_for_analysis.py            # 默认 seed=42，2024-01-01 起
python sample_for_analysis.py --dry-run  # 只打印统计不写文件
python sample_for_analysis.py --seed 7   # 换随机种子抽一组不同样本
```

分组规则：
- **国际游客** = `fromTypeText == "来自Trip.com"` 或 IP 属地在国外白名单（新加坡/日本/马来西亚/泰国/美国/澳大利亚/英国/德国/荷兰/韩国/瑞士/奥地利/尼泊尔/斯里兰卡/加拿大）
- **国内游客** = 其余全部，含内地各省、港澳台、IP 未知/NaN

输出位置：`../panda_analysis/data/panda_sample_2000.csv`

## 字段说明（`ctrip_star*.csv` / `panda_reviews_clean.csv`）

| 字段 | 含义 |
|---|---|
| commentId | 评论唯一 ID（用于断点续爬去重）|
| page | 来自 API 第几页 |
| content | 原始评论文本 |
| content_clean | 清洗后文本（仅 clean.csv 有）|
| score | 评分（1-5 星）|
| publishTime | 原始时间字符串 `/Date(...)/` |
| publishDate | 解析后的可读时间（仅 clean.csv 有）|
| publishTypeTag | 形如 "2026-05-11 发布点评" |
| userNick | 用户昵称（携程已脱敏）|
| userLevel | 用户等级 |
| ipLocatedName | IP 属地（省份）|
| touristTypeDisplay | 出游类型（家庭/情侣/朋友等）|
| usefulCount | 有用数 |
| imageCount / videoCount | 附图、附视频数量 |
| fromTypeText | 评论来源（"来自订单"/"来自Trip.com" 等）|

## Cookie 注意事项

`spider_ctrip.py` 内置一段 Cookie，仅保留访客指纹字段（`UBT_VID`、`GUID`、`_bfa` 等），
**已剔除登录态字段**（`login_uid`/`cticket`/`AHeadUserInfo` 等）。如长期失效，
按以下方式更新：

1. 浏览器打开 https://you.ctrip.com/sight/chengdu104/5414.html
2. F12 → Network → 滑到评论区触发请求
3. 找到 `getCommentCollapseList` 的 fetch 请求
4. 复制 Request Headers 中的 `Cookie`，替换脚本里 `COOKIE` 常量

## 已知限制

- 携程接口分页深度大约 200~1000 页，无法抓全 6.9 万条
- `fromTypeText="来自Trip.com"` 的评论是国际版，文本多为机器翻译
  （会出现"长途车时参观了保护区"这种译文），分析时建议按来源分组或单独剔除
- 评分分布严重偏向 5 星（携程评分本身就有正向偏差）
