"""
Cấu hình chung cho toàn bộ thực nghiệm HPO.
Sửa ở đây, không sửa rải rác trong các file khác -> đảm bảo fair comparison.
"""

# ----- Dataset -----
DATASET_NAME = "wine"   # có thể đổi sang "wine" để thử scalability
TEST_SIZE = 0.2                  # test set bị khóa, chỉ dùng để đánh giá cuối cùng
RANDOM_STATE_SPLIT = 42          # cố định split train/test cho MỌI phương pháp

# ----- Cross-validation (dùng trong lúc search, không đụng tới test set) -----
CV_FOLDS = 5

# ----- Ngân sách đánh giá (fair comparison) -----
N_TRIALS = 60                    # số cấu hình được thử cho MỖI phương pháp

# ----- Seeds để lặp lại thí nghiệm (đánh giá độ ổn định) -----
SEEDS = [0, 1, 2, 3, 4]          # có thể tăng lên 10 nếu đủ thời gian chạy

# ----- Search space cho Random Forest -----
# Dùng chung cho cả Random Search và TPE để so sánh công bằng
SEARCH_SPACE = {
    "n_estimators": (50, 500),          # int
    "max_depth": (3, 30),               # int
    "min_samples_split": (2, 20),       # int
    "max_features": ["sqrt", "log2", None],  # categorical
}

METRIC = "f1_macro"  # metric tối ưu (đổi thành "accuracy" nếu muốn)
