"""
Chạy toàn bộ thực nghiệm: baseline + random_search + tpe, qua nhiều seed.
Kết quả lưu vào results/raw_results.json để Chương 4 dùng vẽ bảng/hình.
Đây chính là "run history" bắt buộc phải nộp (mục 12 trong hướng dẫn).
"""
import json
import os
import config
from baseline import run_baseline
from search_random import run_random_search
from search_tpe import run_tpe_search


def main():
    all_results = []

    for seed in config.SEEDS:
        print(f"\n=== Seed {seed} ===")

        print("Chạy baseline...")
        all_results.append(run_baseline(seed))

        print("Chạy Random Search...")
        all_results.append(run_random_search(seed))

        print("Chạy TPE (Optuna)...")
        all_results.append(run_tpe_search(seed))

    os.makedirs("results", exist_ok=True)
    out_path = "results/raw_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nĐã lưu {len(all_results)} kết quả vào {out_path}")


if __name__ == "__main__":
    main()
