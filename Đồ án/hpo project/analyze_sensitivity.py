"""
Phân tích kết quả sensitivity analysis (results/sensitivity_results.json).

Xuất ra:
1. results/sensitivity_table.csv   -> bảng mean±std theo (method, budget)
2. results/fig_sensitivity.png     -> biểu đồ score trung bình theo budget,
                                       so sánh Random Search vs TPE, có baseline tham chiếu

Chạy: python analyze_sensitivity.py
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

SENSITIVITY_PATH = "results/sensitivity_results.json"
MAIN_RESULTS_PATH = "results/raw_results.json"  # để lấy baseline làm đường tham chiếu
OUT_DIR = "results"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def group_by_method_budget(data):
    groups = {}
    for r in data:
        key = (r["method"], r["budget"])
        groups.setdefault(key, []).append(r)
    return groups


def make_table(groups):
    rows = []
    for (method, budget), records in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        scores = np.array([r["cv_mean"] for r in records])
        times = np.array([r["time_sec"] for r in records])
        rows.append({
            "method": method,
            "budget": budget,
            "n_seeds": len(records),
            "score_mean": scores.mean(),
            "score_std": scores.std(),
            "time_mean": times.mean(),
        })
    return rows


def print_and_save_table(rows, out_path=os.path.join(OUT_DIR, "sensitivity_table.csv")):
    print("\n=== SENSITIVITY THEO BUDGET (Chương 4.8) ===")
    print(f"{'method':15s} {'budget':8s} {'score mean±std':20s} {'time(s)':10s}")
    for r in rows:
        score_str = f"{r['score_mean']:.4f}±{r['score_std']:.4f}"
        print(f"{r['method']:15s} {r['budget']:<8d} {score_str:20s} {r['time_mean']:.1f}")

    header = ["method", "budget", "n_seeds", "score_mean", "score_std", "time_mean"]
    with open(out_path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in header) + "\n")
    print(f"\nĐã lưu {out_path}")


def plot_sensitivity(groups, baseline_mean=None, out_path=os.path.join(OUT_DIR, "fig_sensitivity.png")):
    plt.figure(figsize=(7, 5))

    for method in ["random_search", "tpe"]:
        budgets = sorted(set(b for (m, b) in groups if m == method))
        means = []
        stds = []
        for b in budgets:
            scores = np.array([r["cv_mean"] for r in groups[(method, b)]])
            means.append(scores.mean())
            stds.append(scores.std())
        plt.errorbar(budgets, means, yerr=stds, marker="o", capsize=4, label=method)

    if baseline_mean is not None:
        plt.axhline(baseline_mean, color="gray", linestyle="--", label="baseline")

    plt.xlabel("Budget (số trial)")
    plt.ylabel("Best CV score (mean ± std)")
    plt.title("Sensitivity: ảnh hưởng của budget lên chất lượng nghiệm")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Đã lưu {out_path}")


def get_baseline_mean():
    """Lấy điểm baseline từ thí nghiệm chính (raw_results.json) làm đường tham chiếu,
    vì baseline không phụ thuộc budget nên không cần chạy lại."""
    if not os.path.exists(MAIN_RESULTS_PATH):
        return None
    main_data = load_json(MAIN_RESULTS_PATH)
    baseline_scores = [r["cv_mean"] for r in main_data if r["method"] == "baseline"]
    return np.mean(baseline_scores) if baseline_scores else None


def main():
    data = load_json(SENSITIVITY_PATH)
    groups = group_by_method_budget(data)

    rows = make_table(groups)
    print_and_save_table(rows)

    baseline_mean = get_baseline_mean()
    plot_sensitivity(groups, baseline_mean=baseline_mean)


if __name__ == "__main__":
    main()
