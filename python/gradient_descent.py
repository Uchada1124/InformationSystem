import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 勾配降下法で単回帰の係数を推定する関数
def gradient_descent(x, y, learning_rate=0.01, epochs=10000):
    w = 0  # 傾きの初期値
    w_0 = 0  # 切片の初期値
    n = len(y)

    for _ in range(epochs):
        # 予測値と誤差
        y_pred = w * np.array(x) + w_0
        error = y - y_pred
        
        # 勾配を計算
        w_grad = -2 * np.sum(error * x) / n
        w_0_grad = -2 * np.sum(error) / n
        
        # パラメータの更新
        w -= learning_rate * w_grad
        w_0 -= learning_rate * w_0_grad

    return w, w_0

# 二乗損失を計算する関数
def calculate_squared_loss(x, y, w, b):
    total_loss = sum((yi - (w * xi + b)) ** 2 for xi, yi in zip(x, y))
    return total_loss / len(y)

# プロットする関数
def plot_regression(x, y, w, b, xlabel=None, ylabel=None):
    x_min, x_max = min(x), max(x)
    y_min, y_max = w * x_min + b, w * x_max + b

    plt.scatter(x, y, label="Data Points")
    plt.plot([x_min, x_max], [y_min, y_max], color="red", label="Regression Line")

    # 各点の二乗誤差の線を描画
    for xi, yi in zip(x, y):
        predicted_y = w * xi + b
        plt.plot([xi, xi], [yi, predicted_y], color="gray", linestyle="--")

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.legend()
    plt.title("Linear Regression with Squared Errors")
    plt.show()

def main():
    train_df = pd.read_csv('../data/HousePricesAdvancedRegressionTechniques/train.csv')

    x = train_df['LotArea'][:20].tolist()
    y = train_df['SalePrice'][:20].tolist()

    # 正規化
    x = (np.array(x) - np.mean(x)) / np.std(x)
    y = (np.array(y) - np.mean(y)) / np.std(y)

    # 勾配降下法で回帰係数の推定
    w, w_0 = gradient_descent(x, y)
    print(f"(w, w_0) = ({w}, {w_0})")

    # 二乗損失の計算
    loss = calculate_squared_loss(x, y, w, w_0)
    print("二乗誤差 = {}".format(loss))

    # プロット
    plot_regression(x, y, w, w_0, xlabel='LotArea', ylabel='SalePrice')

if __name__ == '__main__':
    main()
