import numpy as np
import matplotlib.pyplot as plt

import data_utils

def stochastic_gradient_descent_svm(X, y, learning_rate=0.01, epochs=1000):
    """
    ヒンジ損失に基づくSVMの確率的劣勾配降下法

    Args:
        X (ndarray): 特徴量データ (n_samples, n_features)
        y (ndarray): ラベルデータ (n_samples,)
        learning_rate (float): 学習率
        epochs (int): イテレーション数

    Returns:
        w (ndarray): 学習された重み
        b (float): 学習されたバイアス項
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)  # 重みの初期化
    b = 0  # バイアスの初期化

    for epoch in range(epochs):
        # データをシャッフル
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(n_samples):
            xi = X_shuffled[i]
            yi = y_shuffled[i]

            # 判定条件
            if yi * (np.dot(w, xi) + b) < 1:
                # 更新が必要な場合
                w += learning_rate * yi * xi
                b += learning_rate * yi
            else:
                # 更新が不要な場合（w, b をそのまま）
                pass

    return w, b

def plot_svm_decision_boundary(X, y, w, b):
    """
    SVMの学習結果をプロット（決定境界、マージン、サポートベクター）
    
    Args:
        X (ndarray): 特徴量データ (n_samples, n_features)
        y (ndarray): ラベルデータ (n_samples,)
        w (ndarray): 学習された重み
        b (float): 学習されたバイアス項
    """
    plt.figure(figsize=(8, 6))
    
    # ラベルごとにデータをプロット
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1', alpha=0.7, edgecolor='k')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='blue', label='Class -1', alpha=0.7, edgecolor='k')
    
    # 決定境界
    x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_vals = -(w[0] * x_vals + b) / w[1]
    plt.plot(x_vals, y_vals, color='green', label='Decision Boundary')

    # マージン境界
    margin_positive = -(w[0] * x_vals + b - 1) / w[1]
    margin_negative = -(w[0] * x_vals + b + 1) / w[1]
    plt.plot(x_vals, margin_positive, color='green', linestyle='--', label='Margin +1')
    plt.plot(x_vals, margin_negative, color='green', linestyle='--', label='Margin -1')
    
    # サポートベクターのプロット（近似的にマージン内にある点をサポートベクターとみなす）
    support_vector_indices = np.where((y * (np.dot(X, w) + b)) <= 1 + 1e-6)[0]
    plt.scatter(X[support_vector_indices][:, 0], X[support_vector_indices][:, 1],
                s=100, facecolors='none', edgecolors='k', linewidths=1.5, label='Support Vectors')

    # プロットの装飾
    plt.title("SVM Decision Boundary with Margins")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    X, y = data_utils.generate_linear_separable_data(n_samples=200, dim=2, separation=3.0, random_state=42)

    # 学習
    learning_rate = 0.1
    epochs = 1000
    w, b = stochastic_gradient_descent_svm(X, y, learning_rate, epochs)

    print(f"Learned weights: {w}")
    print(f"Learned bias: {b}")

    # 決定境界をプロット
    plot_svm_decision_boundary(X, y, w, b)

if __name__ == '__main__':
    main()