"""
Sensitivity analysis theo BUDGET (số trial cho phép).
Trả lời câu hỏi: "Nếu tăng/giảm budget, kết quả thay đổi thế nào?"
(mục 4.8 trong dàn ý báo cáo, và là câu hỏi hay gặp khi bảo vệ - mục 34).

Cách làm: với mỗi budget trong BUDGETS, chạy lại Random Search và TPE
qua các seed trong SENSITIVITY_SEEDS (dùng ít seed hơn thí nghiệm chính
để tiết kiệm thời gian - sẽ nêu rõ trong báo cáo đây là đánh đổi hợp lý).

Baseline KHÔNG phụ thuộc budget (không search) nên không cần chạy lại.

Chạy: python run_sensitivity.py
Kết quả lưu: results/sensitivity_results.json
"""
import json
import os
import time

import config
from search_random import run_random_search
from search_tpe import run_tpe_search

# ----- Cấu hình riêng cho sensitivity analysis -----
BUDGETS = [30, 60, 100]          # số trial để so sánh
SENSITIVITY_SEEDS = [0, 1, 2]    # dùng ít seed hơn (3 thay vì 5) để tiết kiệm thời gian


def main():
    all_results = []
    t_start = time.time()

    for budget in BUDGETS:
        config.N_TRIALS = budget  # override budget cho các hàm search dùng config.N_TRIALS

        for seed in SENSITIVITY_SEEDS:
            print(f"\n=== budget={budget}, seed={seed} ===")

            print("Random Search...")
            r_rs = run_random_search(seed)
            r_rs["budget"] = budget
            all_results.append(r_rs)

            print("TPE...")
            r_tpe = run_tpe_search(seed)
            r_tpe["budget"] = budget
            all_results.append(r_tpe)

    os.makedirs("results", exist_ok=True)
    out_path = "results/sensitivity_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nTổng thời gian: {time.time() - t_start:.1f}s")
    print(f"Đã lưu {len(all_results)} kết quả vào {out_path}")


if __name__ == "__main__":
    main()
