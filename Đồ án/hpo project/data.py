"""
Load dataset và chia train/test.
Test set chỉ được dùng ở bước đánh giá cuối (evaluate.py), KHÔNG dùng khi search.
"""
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
import config


def load_data():
    if config.DATASET_NAME == "breast_cancer":
        data = load_breast_cancer()
    elif config.DATASET_NAME == "wine":
        data = load_wine()
    else:
        raise ValueError(f"Dataset không hỗ trợ: {config.DATASET_NAME}")

    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE_SPLIT,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Số lớp: {len(set(y_train))}")
