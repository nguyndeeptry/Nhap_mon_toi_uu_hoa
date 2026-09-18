# HPO cho Random Forest — Đồ án Nhập môn Tối ưu hóa

So sánh Baseline (default) vs Random Search vs Bayesian Optimization (TPE/Optuna)
trong bài toán tối ưu siêu tham số cho mô hình phân loại.

## Cách chạy

```bash
pip install scikit-learn optuna numpy matplotlib --break-system-packages

python data.py             # kiểm tra dữ liệu load đúng
python baseline.py         # chạy thử baseline
python search_random.py    # chạy thử random search
python search_tpe.py       # chạy thử TPE

python run_experiment.py   # CHẠY TOÀN BỘ (baseline + random + tpe) x nhiều seed
```

Kết quả full lưu ở `results/raw_results.json`.

## Đã kiểm tra (sanity check)

Đã chạy thử với N_TRIALS=5, SEEDS=[0,1] — pipeline chạy sạch, không lỗi,
kết quả hợp lý (TPE ≥ Random Search ≥ Baseline).

## Việc cần làm tiếp theo (theo đúng thứ tự)

1. **Chỉnh `config.py`**: cân nhắc đổi N_TRIALS lên 60, SEEDS lên 5-10 seed
   tùy thời gian máy chạy được (chạy thử full 1 lần trước để đo thời gian
   thực tế, nhân lên để ước lượng tổng thời gian).
2. **Chạy full**: `python run_experiment.py` (có thể chạy qua đêm nếu N_TRIALS=60,
   SEEDS=10 → 30 lần chạy CV, mỗi lần vài chục trial).
3. **Viết `analyze.py`** để đọc `results/raw_results.json` và:
   - Tính mean ± std của cv_mean cho mỗi method qua các seed → bảng Chương 4.4
   - Vẽ convergence curve: score tốt nhất theo từng trial (từ `trial_history`)
     của Random Search vs TPE → so sánh tốc độ hội tụ.
   - Vẽ boxplot cv_mean qua các seed cho mỗi method → thể hiện độ ổn định
     (mục 11 trong hướng dẫn: không được chỉ báo 1 con số).
4. **Đánh giá cuối cùng trên test set** (chỉ 1 lần, với best_params của mỗi
   phương pháp) — đây là số liệu "thật" để báo cáo, KHÔNG dùng cv_mean của
   tập train để kết luận cuối.
5. **Sensitivity analysis**: thử lại với N_TRIALS = 30 và 100 xem TPE có lợi
   thế rõ hơn khi budget nhỏ không (đây là nội dung mục 4.8 trong dàn ý báo cáo).
6. **(Tùy chọn) Scalability**: đổi `DATASET_NAME = "wine"` chạy lại, xem
   thời gian/kết quả thay đổi thế nào khi dữ liệu khác.
7. Viết báo cáo theo 5 chương, dùng đúng số liệu từ `results/raw_results.json`
   — không tự chế số liệu, tránh lỗi "số liệu không khớp code/log" (mục 24).

## Checklist tránh lỗi thường gặp (theo hướng dẫn đồ án)

- [ ] Test set (X_test, y_test) tuyệt đối không dùng trong lúc search
- [ ] Baseline, Random Search, TPE dùng CÙNG search space, CÙNG N_TRIALS
- [ ] Chạy tối thiểu vài seed, báo cáo mean ± std, không chọn "best run"
- [ ] Lưu trial_history để vẽ convergence, không chỉ báo con số cuối
- [ ] Giải thích rõ TPE hoạt động thế nào (không chỉ gọi Optuna mà không hiểu)
