# -*- coding: utf-8 -*-
"""
run_all.py — 一键跑全部分析流程
=================================
按顺序执行阶段 3 → 1 → 2 → 4 → 报告 → PPT。
每步前检查前置依赖是否存在,失败时打印明确提示并中止。

用法:
    python run_all.py                # 全量(含 Gemini API 调用)
    python run_all.py --skip-api     # 跳过阶段3/4 的 API 调用,只基于已有结果出图表/报告
    python run_all.py --from 3       # 从阶段3 开始,随后重跑依赖它的阶段1/2/4

注意:
    - 阶段 3/4 需要 Gemini API key(配置在 .env 里)
    - 阶段 3 全量约消耗 ~50k tokens,耗时 2-5 分钟
    - 阶段 1 会使用阶段 3 的 is_translated 标记过滤机翻伪影,
      所以阶段 3 必须先于阶段 1 执行。
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent

STAGES = [
    {
        "name": "阶段3:情感分析",
        "script": "03_sentiment.py",
        "check_before": [Path("data/panda_sample_2000.csv")],
        "check_after": [Path("output/3_sentiment/sentiment_results.csv")],
        "is_api": True,
        "skip_api_args": ["--analyze-only"],
        "stage_no": 3,
    },
    {
        "name": "阶段1:词频与词云",
        "script": "01_word_freq.py",
        "check_before": [Path("data/panda_sample_2000.csv")],
        "check_after": [Path("output/1_word_freq/all_top50.csv")],
        "stage_no": 1,
    },
    {
        "name": "阶段2:社会语义网络",
        "script": "02_network.py",
        "check_before": [Path("output/1_word_freq/all_top50.csv")],
        "check_after": [Path("output/2_network/centrality_all.csv")],
        "stage_no": 2,
    },
    {
        "name": "阶段4:主题维度提炼",
        "script": "04_themes.py",
        "check_before": [
            Path("output/1_word_freq/all_top50.csv"),
            Path("output/2_network/centrality_all.csv"),
            Path("output/3_sentiment/sentiment_results.csv"),
        ],
        "check_after": [Path("output/4_themes/full_analysis_report.md")],
        "is_api": True,
        "skip_api_args": ["--offline"],
        "stage_no": 4,
    },
    {
        "name": "生成 Word 报告",
        "script": "make_report.py",
        "check_before": [Path("output/3_sentiment/sentiment_results.csv")],
        "check_after": [],
        "stage_no": 5,
    },
    {
        "name": "生成演示 PPT",
        "script": "make_ppt.py",
        "check_before": [],
        "check_after": [],
        "stage_no": 6,
    },
]


def run_stage(stage: dict, skip_api: bool) -> bool:
    """运行一个阶段,返回是否成功。"""
    name = stage["name"]
    script = stage["script"]
    is_api = stage.get("is_api", False)

    print(f"\n{'='*60}")
    print(f"▶ {name}  ({script})")
    print("=" * 60)

    # 前置检查
    for dep in stage.get("check_before", []):
        if not (HERE / dep).exists():
            print(f"  ❌ 前置依赖缺失: {dep}")
            print(f"     请先确保前面的阶段已成功运行")
            return False

    # 构造命令
    cmd = [sys.executable, str(HERE / script)]
    if is_api and skip_api:
        args = stage.get("skip_api_args", [])
        cmd.extend(args)
        print(f"  (skip-api 模式,追加参数: {' '.join(args)})")

    # 执行
    result = subprocess.run(cmd, cwd=str(HERE))
    if result.returncode != 0:
        print(f"\n  ❌ {name} 失败 (exit code {result.returncode})")
        return False

    # 后置检查
    for out in stage.get("check_after", []):
        if not (HERE / out).exists():
            print(f"  ⚠️  预期产出 {out} 未生成,但脚本未报错")

    print(f"  ✅ {name} 完成")
    return True


def main():
    ap = argparse.ArgumentParser(description="一键跑全部分析流程")
    ap.add_argument("--skip-api", action="store_true",
                    help="跳过 Gemini API 调用(阶段3 用 --analyze-only,阶段4 用 --offline)")
    ap.add_argument("--from", type=int, default=1, dest="start_from",
                    help="从第几阶段开始(1-6),默认 1。若从 1 开始,会先跑阶段3以供阶段1过滤机翻。")
    args = ap.parse_args()

    if args.start_from <= 1:
        stages_to_run = STAGES
    elif args.start_from == 3:
        # 阶段3结果会改变阶段1的机翻过滤,因此继续重跑1/2/4。
        stages_to_run = STAGES
    elif args.start_from == 2:
        stages_to_run = [s for s in STAGES if s["stage_no"] in {2, 4, 5, 6}]
    else:
        stages_to_run = [s for s in STAGES if s["stage_no"] >= args.start_from]

    print("🚀 成都大熊猫基地评论分析 — 全流程执行")
    print(f"   共 {len(stages_to_run)} 步,从「{stages_to_run[0]['name']}」开始")
    if args.skip_api:
        print("   ⚡ skip-api 模式:跳过 Gemini 调用")

    for stage in stages_to_run:
        ok = run_stage(stage, skip_api=args.skip_api)
        if not ok:
            print(f"\n🛑 流程中止于「{stage['name']}」")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("🎉 全部流程完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
