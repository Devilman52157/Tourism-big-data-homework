# 成都大熊猫繁育研究基地游客评论文本挖掘

对 2000 条携程评论做四类分析,聚焦 **国内游客 vs 国际游客** 的对比:
1. 词频分析 + 词云
2. 社会语义网络
3. 情感分析(Gemini API)
4. 主题维度提炼(Gemini API)

---

## 一、环境要求

- Windows / macOS / Linux
- Python 3.10 或以上(本项目用 3.12 测试)
- 推荐使用 [`uv`](https://github.com/astral-sh/uv) 管理虚拟环境(更快)。没有也可用 `venv`。

## 二、第一次运行(环境复刻)

### 1. 克隆/拷贝项目后,进入项目目录

Windows PowerShell:
```powershell
cd D:\Desktop\旅游作业\panda_analysis
```

### 2. 创建虚拟环境

**推荐用 uv:**
```powershell
pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple
uv venv .venv --python 3.12
```

**没有 uv 也行,用 venv:**
```powershell
python -m venv .venv
```

### 3. 激活虚拟环境

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```
> 若 PowerShell 报"无法加载脚本"安全错误,运行一次:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

Windows CMD:
```cmd
.venv\Scripts\activate.bat
```

macOS / Linux:
```bash
source .venv/bin/activate
```

激活成功后,命令行最前面会出现 `(.venv)` 字样。

### 4. 安装依赖

用 uv 安装(推荐,十几秒就装完):
```powershell
uv pip install -r requirements.txt
```

或用普通 pip(可能慢一点,加镜像):
```powershell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 5. 配置 Gemini API Key

复制 `.env.example` 为 `.env`,把 `your_api_key_here` 改成你自己的真实 key:
```powershell
Copy-Item .env.example .env
notepad .env
```

API key 在 [Google AI Studio](https://aistudio.google.com/) 免费申请。

### 6. 准备数据

分析项目的数据来自爬虫项目的产出。如果你是第一次运行,或想复刻/更新样本,先到 `panda_spider/` 目录下跑:

```powershell
cd ..\panda_spider
.\.venv\Scripts\Activate.ps1   # 激活爬虫项目的虚拟环境(二者独立)
python spider_ctrip.py --star 1 ...   # 详见 panda_spider/README.md
python clean_data.py
python sample_for_analysis.py   # ★ 抽出 2000 条样本写到 panda_analysis/data/
```

最后一步 `sample_for_analysis.py` 会:
1. 读 `panda_spider/data/panda_reviews_clean.csv`;
2. 按"`fromTypeText == 来自Trip.com` 或 IP 属地在国外白名单 → 国际游客,其余 → 国内游客"打 group 标签;
3. 按 score(1–5 星)分层抽样国内 1000 + 国际 1000,默认只保留 2024-01-01 之后的评论;
   该样本用于保证不同评分层都有足够文本可比,不是自然总体比例;
4. 二次检查是否有明显酒店/机场/商圈离题评论,有则中止;
5. 输出到 `panda_analysis/data/panda_sample_2000.csv`(UTF-8-sig)。

情感分析阶段会另外输出 `output/3_sentiment/sentiment_weighted_distribution.csv`
和同名 PNG,按清洗后总体的 `group × score` 分布校正整体口碑比例。报告中的
“分层样本情感分布”用于组间比较,“加权情感分布”用于总体描述。

如果你只想直接复现分析,把已有的 `panda_sample_2000.csv` 放到 `data/` 目录下即可。

---

## 三、运行各阶段脚本

> 每次开新窗口都要先 **激活虚拟环境**(见上面第 3 步)。

推荐直接运行 `python run_all.py`。全流程会先跑阶段 3 生成 `is_translated`
标记,再跑阶段 1/2/4;这样词频阶段可以过滤机翻伪影,不会因为首次运行时
缺少情感结果而产生不同口径。

| 阶段 | 命令 | 作用 |
|------|------|------|
| 字体测试 | `python font_config.py` | 检测中文字体,生成测试图 |
| 阶段 3 | `python 03_sentiment.py` | 情感分析(消耗 API 配额,先运行) |
| 阶段 1 | `python 01_word_freq.py` | 词频 + 词云(使用阶段 3 的机翻标记) |
| 阶段 2 | `python 02_network.py` | 社会语义网络 |
| 阶段 4 | `python 04_themes.py` | AI 主题提炼(消耗 API 配额) |

> 若只想基于已有 API 结果重出图表,可运行 `python run_all.py --skip-api`。

---

## 四、目录结构

```
panda_analysis/
├── data/                       # 原始数据(不入库)
├── output/                     # 各阶段输出图表(不入库)
│   ├── 1_word_freq/
│   ├── 2_network/
│   ├── 3_sentiment/
│   └── 4_themes/
├── stopwords.txt               # 中文停用词表
├── custom_dict.txt             # jieba 自定义词典
├── font_config.py              # 中文字体配置模块
├── .env                        # API key(本地,不入库)
├── .env.example                # API key 模板(可分享)
├── .gitignore
├── requirements.txt
├── 01_word_freq.py             # 阶段 1：词频与词云
├── 02_network.py               # 阶段 2：社会语义网络
├── 03_sentiment.py             # 阶段 3：Gemini 情感与方面分析
├── 04_themes.py                # 阶段 4：主题维度提炼与综合报告
├── make_report.py              # 生成 Word 实验报告
├── make_ppt.py                 # 生成演示 PPT
└── README.md
```

---

## 五、常见问题

**Q1. 激活虚拟环境时 PowerShell 报"无法运行脚本"**
运行 `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`,然后重试。

**Q2. matplotlib 中文显示成方块**
- Windows:确保系统装了"微软雅黑"(默认就有)。
- macOS:`PingFang SC` 默认存在。
- Linux:`sudo apt-get install fonts-noto-cjk`。
- 检测方式:`python font_config.py`。

**Q3. pip 装包很慢或失败**
加清华镜像:`pip install xxx -i https://pypi.tuna.tsinghua.edu.cn/simple`。
其他可选镜像:阿里 `https://mirrors.aliyun.com/pypi/simple/`、豆瓣 `https://pypi.douban.com/simple`。

**Q4. `wordcloud` 安装失败**
报"Microsoft Visual C++ 14.0 required"时,先安装 [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/),或用清华镜像走预编译 wheel。

**Q5. Gemini API 报 429 / 配额超限**
免费版有速率限制。脚本会自动 sleep 重试;实在不行换更慢的速率或申请付费 key。

**Q6. 读 CSV 报 `UnicodeDecodeError`**
本数据是 UTF-8 with BOM,务必用 `encoding="utf-8-sig"` 读。

---

## 六、数据字段说明

只用以下列做分析,其他列忽略:

| 列名 | 含义 |
|------|------|
| `content_clean` | 已清洗的评论文本(主分析字段) |
| `score` | 评分 1–5 星 |
| `publishDate` | 发布时间 |
| `group` | `国内游客` / `国际游客`(对比核心) |
| `touristTypeDisplay` | 家庭/情侣/朋友/单独 |
| `ipLocatedName` | IP 属地 |
| `userNick` | 昵称(备用) |
