import argparse
from pathlib import Path

from src.inference.run_forecast import run_forecast
from src.backtest.run_backtest import run_backtest


def parse_cutoff_days(cutoff_day: int | None, cutoff_days: str | None):
    if cutoff_days:
        values = []
        for x in cutoff_days.split(','):
            x = x.strip()
            if not x:
                continue
            values.append(int(x))
        return tuple(values)
    if cutoff_day is not None:
        return (int(cutoff_day),)
    return (0, 5, 10, 15, 20, 25)


def main():
    parser = argparse.ArgumentParser(description="Ecommerce sales forecast project entrypoint")
    parser.add_argument("--action", default="status", choices=["status", "init", "forecast", "backtest"], help="Action to run")
    parser.add_argument("--target-month", default="2026-03", help="目标月份，格式 YYYY-MM")
    parser.add_argument("--cutoff-day", type=int, default=None, help="单个截止日。0 表示截止上个月预测下月整月")
    parser.add_argument("--cutoff-days", type=str, default=None, help="多个截止日，逗号分隔，例如 0,7,14,21,28")
    parser.add_argument("--brand", type=str, default="total", help="品牌范围：具体品牌名 / total / all")
    parser.add_argument("--platform", type=str, default="total", help="平台范围：具体平台名 / total / all")
    args = parser.parse_args()

    if args.action == "status":
        print("sales_forecast 项目骨架已就绪")
        print(f"项目目录: {Path(__file__).resolve().parent}")
    elif args.action == "init":
        print("初始化动作预留：后续可加入数据检查、目录检查、配置校验")
    elif args.action == "forecast":
        cutoff = 15 if args.cutoff_day is None else args.cutoff_day
        result = run_forecast(target_month=args.target_month, cutoff_day=cutoff, brand=args.brand, platform=args.platform)
        print(result.to_string(index=False))
    elif args.action == "backtest":
        cutoff_days = parse_cutoff_days(args.cutoff_day, args.cutoff_days)
        result_df, summary = run_backtest(
            cutoff_days=cutoff_days,
            exclude_months=["2026-03"],
            brand=args.brand,
            platform=args.platform,
        )
        print("=== BACKTEST SUMMARY ===")
        print(summary.to_string(index=False))
        print("\n输出文件:")
        print("- data/backtest_result/backtest_v1.csv")
        print("- data/backtest_result/backtest_v1_summary.csv")


if __name__ == "__main__":
    main()
