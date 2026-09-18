"""
Đánh giá CUỐI CÙNG trên test set - chỉ dùng đúng 1 lần, sau khi đã chọn xong
hyperparameter bằng CV trên train set (search_random.py, search_tpe.py).

Đây là số liệu "thật" để đưa vào Chương 4 / báo cáo, KHÔNG phải cv_mean.
cv_mean chỉ dùng để CHỌN tham số; test set dùng để ĐÁNH GIÁ tham số đã chọn.

Với mỗi seed, mỗi method:
  1. Lấy best_params đã tìm được (từ raw_results.json)
  2. Huấn luyện lại RandomForest trên TOÀN BỘ train set với best_params đó
  3. Predict trên test set -> tính accuracy, f1_macro
  4. Lưu lại, sau đó tổng hợp mean±std qua các seed cho mỗi method

Chạy: python evaluate_test.py
"""
import json
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from data import load_data

RESULTS_PATH = "results/raw_results.json"
OUT_DIR = "results"

# Các tham số của RandomForestClassifier mà ta thực sự tune
# (baseline có nhiều key khác trong best_params, cần lọc lại cho đúng)
TUNED_KEYS = ["n_estimators", "max_depth", "min_samples_split", "max_features"]


def load_results(path=RESULTS_PATH):
    with open(path) as f:
        return json.load(f)


def clean_params(best_params):
    """Chỉ giữ lại đúng các hyperparameter nằm trong search space,
    tránh lỗi khi best_params của baseline chứa nhiều key thừa của sklearn."""
    return {k: best_params[k] for k in TUNED_KEYS if k in best_params}


def evaluate_one(record, X_train, y_train, X_test, y_test):
    params = clean_params(record["best_params"])
    seed = record["seed"]

    model = RandomForestClassifier(random_state=seed, **params)
    model.fit(X_train, y_train)          # train trên TOÀN BỘ train set (không CV nữa)
    y_pred = model.predict(X_test)       # predict trên test set - CHỈ 1 LẦN

    return {
        "method": record["method"],
        "seed": seed,
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "params_used": params,
    }


def summarize(test_results):
    methods = sorted(set(r["method"] for r in test_results))
    summary = []
    for m in methods:
        rows = [r for r in test_results if r["method"] == m]
        acc = np.array([r["test_accuracy"] for r in rows])
        f1 = np.array([r["test_f1_macro"] for r in rows])
        summary.append({
            "method": m,
            "n_seeds": len(rows),
            "test_acc_mean": acc.mean(),
            "test_acc_std": acc.std(),
            "test_f1_mean": f1.mean(),
            "test_f1_std": f1.std(),
        })
    return summary


def print_and_save(summary, out_path=os.path.join(OUT_DIR, "test_summary.csv")):
    print("\n=== KẾT QUẢ TRÊN TEST SET (số liệu chính thức cho Chương 4) ===")
    print(f"{'method':15s} {'accuracy':20s} {'f1_macro':20s}")
    for r in summary:
        acc_str = f"{r['test_acc_mean']:.4f}±{r['test_acc_std']:.4f}"
        f1_str = f"{r['test_f1_mean']:.4f}±{r['test_f1_std']:.4f}"
        print(f"{r['method']:15s} {acc_str:20s} {f1_str:20s}")

    header = ["method", "n_seeds", "test_acc_mean", "test_acc_std", "test_f1_mean", "test_f1_std"]
    with open(out_path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in summary:
            f.write(",".join(str(r[h]) for h in header) + "\n")
    print(f"\nĐã lưu {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data()   # cùng 1 split cố định như lúc search
    records = load_results()

    test_results = [evaluate_one(r, X_train, y_train, X_test, y_test) for r in records]

    with open(os.path.join(OUT_DIR, "test_results_detail.json"), "w") as f:
        json.dump(test_results, f, indent=2)

    summary = summarize(test_results)
    print_and_save(summary)


if __name__ == "__main__":
    main()
