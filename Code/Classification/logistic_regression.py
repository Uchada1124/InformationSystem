import numpy as np
import matplotlib.pyplot as plt

import data_utils

def sigmoid(z):
    """シグモイド関数"""
    return 1 / (1 + np.exp(-z))

def stochastic_gradient_descent_logistic(X, y, learning_rate=0.01, epochs=1000):
    """
    純粋な確率的勾配降下法でロジスティック回帰を最適化

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
            xi = X_shuffled[i].reshape(1, -1)  # データ点
            yi = y_shuffled[i]  # ラベル
            
            # 勾配の計算
            z = np.dot(xi, w) + b
            v = np.exp(-yi * z)
            grad_w = -(yi * xi.flatten() * v / (1 + v))
            grad_b = -(yi * v / (1 + v))
            
            # パラメータの更新
            w -= learning_rate * grad_w
            b -= learning_rate * grad_b

    return w, b

def plot_decision_boundary(X, y, w, b):
    """
    学習されたモデルの決定境界をプロット

    Args:
        X (ndarray): 特徴量データ
        y (ndarray): ラベルデータ
        w (ndarray): 学習された重み
        b (float): 学習されたバイアス項
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1', alpha=0.7, edgecolor='k')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='blue', label='Class 0', alpha=0.7, edgecolor='k')
    
    # 決定境界
    x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_vals = -(w[0] * x_vals + b) / w[1]
    plt.plot(x_vals, y_vals, color='green', label='Decision Boundary')
    
    plt.title("Logistic Regression with SGD")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # 線形分離可能なデータを生成
    X, y = data_utils.generate_linear_separable_data(n_samples=200, dim=2, separation=3.0, random_state=42)

    # 学習
    learning_rate = 0.1
    epochs = 1000
    w, b = stochastic_gradient_descent_logistic(X, y, learning_rate, epochs)

    print(f"Learned weights: {w}")
    print(f"Learned bias: {b}")

    # 決定境界をプロット
    plot_decision_boundary(X, y, w, b)

    # 線形分離可能なデータを生成
    X, y = data_utils.generate_non_linear_separable_data(n_samples=200, dim=2, noise=0.5, random_state=42)

    # 学習
    learning_rate = 0.1
    epochs = 1000
    w, b = stochastic_gradient_descent_logistic(X, y, learning_rate, epochs)

    print(f"Learned weights: {w}")
    print(f"Learned bias: {b}")

    # 決定境界をプロット
    plot_decision_boundary(X, y, w, b)

if __name__ == "__main__":
    main()
