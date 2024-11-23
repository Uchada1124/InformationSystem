import numpy as np
import matplotlib.pyplot as plt

def generate_linear_separable_data(n_samples=100, dim=2, separation=2.0, random_state=None):
    """
    線形分離可能なデータを生成する関数。

    Args:
        n_samples (int): サンプル数（各クラスのサンプル数は n_samples / 2）。
        dim (int): 特徴量の次元数。
        separation (float): クラス間の分離度合い（大きいほど分離しやすい）。
        random_state (int): ランダムシード（Noneの場合はランダム）。
    
    Returns:
        X (ndarray): 特徴量データ (n_samples, dim)。
        y (ndarray): ラベルデータ (n_samples,)。
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # 各クラスのサンプル数を計算
    n_class_samples = n_samples // 2

    # クラス1のデータを生成
    class_1 = np.random.randn(n_class_samples, dim) + separation

    # クラス0のデータを生成
    class_0 = np.random.randn(n_class_samples, dim) - separation

    # 特徴量データを結合
    X = np.vstack((class_1, class_0))
    
    # ラベルデータを生成
    y = np.hstack((np.ones(n_class_samples), -np.ones(n_class_samples)))

    return X, y

def generate_non_linear_separable_data(n_samples=100, dim=2, noise=0.5, random_state=None):
    """
    線形分離不可能なデータを生成する関数。

    Args:
        n_samples (int): サンプル数（各クラスのサンプル数は n_samples / 2）。
        dim (int): 特徴量の次元数。
        noise (float): データに加えるノイズの大きさ（大きいほど重なりが増加）。
        random_state (int): ランダムシード（Noneの場合はランダム）。
    
    Returns:
        X (ndarray): 特徴量データ (n_samples, dim)。
        y (ndarray): ラベルデータ (n_samples,)。
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # 各クラスのサンプル数を計算
    n_class_samples = n_samples // 2

    # クラス1のデータを生成
    class_1 = np.random.randn(n_class_samples, dim) + 2.0  # 中心を +2 にシフト
    # クラス0のデータを生成
    class_0 = np.random.randn(n_class_samples, dim) - 2.0  # 中心を -2 にシフト

    # ノイズを追加
    class_1 += noise * np.random.randn(n_class_samples, dim)
    class_0 += noise * np.random.randn(n_class_samples, dim)

    # 特徴量データを結合
    X = np.vstack((class_1, class_0))
    
    # ラベルデータを生成
    y = np.hstack((np.ones(n_class_samples), -np.ones(n_class_samples)))

    return X, y

def plot_generated_data(X, y):
    """
    データをプロットする関数。

    Args:
        X (ndarray): 特徴量データ。
        y (ndarray): ラベルデータ。
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1', alpha=0.7, edgecolor='k')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='blue', label='Class 0', alpha=0.7, edgecolor='k')
    plt.title("Generated Linear Separable Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # データを生成
    X, y = generate_linear_separable_data(n_samples=200, dim=2, separation=3.0, random_state=42)

    # データをプロット
    plot_generated_data(X, y)

    X, y = generate_non_linear_separable_data(n_samples=200, dim=2, noise=0.5, random_state=42)

    # データをプロット
    plot_generated_data(X, y)

if __name__ == "__main__":
    main()
