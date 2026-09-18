"""
Phân tích kết quả từ results/raw_results.json.

Xuất ra:
1. results/summary_table.csv       -> bảng mean/std cv_mean và time cho mỗi method (Chương 4.4, 4.5)
2. results/fig_convergence.png     -> đường hội tụ trung bình qua các seed (Chương 4.4)
3. results/fig_boxplot.png         -> phân phối cv_mean qua các seed (Chương 4 / mục 11: không báo 1 con số)
4. results/fig_boxplot_time.png    -> phân phối thời gian chạy qua các seed (Chương 4.5)

Chạy: python analyze.py
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_PATH = "results/raw_results.json"
OUT_DIR = "results"


def load_results(path=RESULTS_PATH):
    with open(path) as f:
        data = json.load(f)
    return data


def group_by_method(data):
    """{'baseline': [...], 'random_search': [...], 'tpe': [...]}"""
    groups = {}
    for r in data:
        groups.setdefault(r["method"], []).append(r)
    return groups


# ---------------------------------------------------------------------------
# 1. Bảng tổng hợp mean ± std
# ---------------------------------------------------------------------------
def make_summary_table(groups):
    rows = []
    for method, records in groups.items():
        cv_means = np.array([r["cv_mean"] for r in records])
        times = np.array([r["time_sec"] for r in records])
        rows.append({
            "method": method,
            "n_seeds": len(records),
            "cv_mean_avg": cv_means.mean(),
            "cv_mean_std": cv_means.std(),
            "cv_mean_best": cv_means.max(),
            "cv_mean_worst": cv_means.min(),
            "time_avg_sec": times.mean(),
            "time_std_sec": times.std(),
        })
    return rows


def print_and_save_table(rows, out_path=os.path.join(OUT_DIR, "summary_table.csv")):
    header = ["method", "n_seeds", "cv_mean_avg", "cv_mean_std",
              "cv_mean_best", "cv_mean_worst", "time_avg_sec", "time_std_sec"]

    print("\n=== BẢNG TỔNG HỢP (Chương 4.4 / 4.5) ===")
    print(f"{'method':15s} {'mean±std':16s} {'best':8s} {'worst':8s} {'time(s) mean±std':20s}")
    for r in rows:
        mean_std = f"{r['cv_mean_avg']:.4f}±{r['cv_mean_std']:.4f}"
        time_str = f"{r['time_avg_sec']:.2f}±{r['time_std_sec']:.2f}"
        print(f"{r['method']:15s} {mean_std:16s} {r['cv_mean_best']:.4f}  {r['cv_mean_worst']:.4f}  {time_str}")

    with open(out_path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in header) + "\n")
    print(f"\nĐã lưu bảng vào {out_path}")


# ---------------------------------------------------------------------------
# 2. Convergence curve: best-so-far trung bình qua các seed, theo từng trial
# ---------------------------------------------------------------------------
def best_so_far(trial_history):
    """Từ list [{'trial':i,'score':s},...] -> mảng best score tính đến trial i."""
    scores = [t["score"] for t in sorted(trial_history, key=lambda x: x["trial"])]
    best = -np.inf
    curve = []
    for s in scores:
        best = max(best, s)
        curve.append(best)
    return np.array(curve)


def plot_convergence(groups, out_path=os.path.join(OUT_DIR, "fig_convergence.png")):
    plt.figure(figsize=(7, 5))

    for method in ["random_search", "tpe"]:  # baseline không có trial_history (chỉ 1 lần chạy)
        records = groups.get(method, [])
        if not records or "trial_history" not in records[0]:
            continue
        curves = np.stack([best_so_far(r["trial_history"]) for r in records])  # (n_seeds, n_trials)
        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0)
        x = np.arange(1, len(mean_curve) + 1)

        plt.plot(x, mean_curve, label=method)
        plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

    # đường baseline nằm ngang để so sánh
    if "baseline" in groups:
        baseline_mean = np.mean([r["cv_mean"] for r in groups["baseline"]])
        plt.axhline(baseline_mean, color="gray", linestyle="--", label="baseline")

    plt.xlabel("Số trial")
    plt.ylabel("Best CV score đến hiện tại")
    plt.title("Convergence: Random Search vs TPE (mean ± std qua các seed)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Đã lưu {out_path}")


# ---------------------------------------------------------------------------
# 3. Boxplot: phân phối kết quả qua các seed (không được chỉ báo 1 số - mục 11)
# ---------------------------------------------------------------------------
def plot_boxplot(groups, out_path=os.path.join(OUT_DIR, "fig_boxplot.png")):
    methods = list(groups.keys())
    data_per_method = [[r["cv_mean"] for r in groups[m]] for m in methods]

    plt.figure(figsize=(6, 5))
    plt.boxplot(data_per_method, tick_labels=methods, showmeans=True)
    plt.ylabel(f"CV score")
    plt.title("Độ ổn định qua các seed (boxplot)")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Đã lưu {out_path}")


def plot_boxplot_time(groups, out_path=os.path.join(OUT_DIR, "fig_boxplot_time.png")):
    methods = list(groups.keys())
    data_per_method = [[r["time_sec"] for r in groups[m]] for m in methods]

    plt.figure(figsize=(6, 5))
    plt.boxplot(data_per_method, tick_labels=methods, showmeans=True)
    plt.ylabel("Thời gian chạy (giây)")
    plt.title("Hiệu quả: thời gian chạy qua các seed")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Đã lưu {out_path}")


# ---------------------------------------------------------------------------
# 4. So sánh best_params giữa các seed (để thảo luận độ nhạy / tính nhất quán)
# ---------------------------------------------------------------------------
def print_best_params_variability(groups):
    print("\n=== BEST PARAMS QUA CÁC SEED (tham khảo cho phần Discussion) ===")
    for method in ["random_search", "tpe"]:
        if method not in groups:
            continue
        print(f"\n{method}:")
        for r in groups[method]:
            bp = r["best_params"]
            print(f"  seed={r['seed']}: n_estimators={bp.get('n_estimators')}, "
                  f"max_depth={bp.get('max_depth')}, "
                  f"min_samples_split={bp.get('min_samples_split')}, "
                  f"max_features={bp.get('max_features')}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_results()
    groups = group_by_method(data)

    rows = make_summary_table(groups)
    print_and_save_table(rows)

    plot_convergence(groups)
    plot_boxplot(groups)
    plot_boxplot_time(groups)

    print_best_params_variability(groups)

    print("\nXong. Các file trong results/ dùng trực tiếp cho Chương 4 của báo cáo.")


if __name__ == "__main__":
    main()
