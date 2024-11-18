import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 単回帰を計算する関数
def calculate_simple_regression_weights(x, y):
    tmp = sum((xi - np.mean(x)) * (yi - np.mean(y)) for xi, yi in zip(x, y))
    w = tmp / (np.std(x) ** 2 * len(x))
    w_0 = np.mean(y) - (w * np.mean(x))
    return w, w_0

# 二乗損失を計算する関数
def calculate_simple_regression_squared_loss(x, y, w, b):
    total_loss = sum((yi - (w * xi + b)) ** 2 for xi, yi in zip(x, y))
    return total_loss / len(y)

# プロットする関数
def plot_simple_regression(x, y, w, b, xlabel=None, ylabel=None):
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
    train_df = pd.read_csv('./data/HousePricesAdvancedRegressionTechniques/train.csv')

    x = train_df['LotArea'][:20].tolist()
    y = train_df['SalePrice'][:20].tolist()

    # x = train_df['LotArea'].tolist()
    # y = train_df['SalePrice'].tolist()

    # 標準化
    x = (np.array(x) - np.mean(x)) / np.std(x)
    y = (np.array(y) - np.mean(y)) / np.std(y)

    # 回帰係数の計算
    w, w_0 = calculate_simple_regression_weights(x, y)
    print(f"(w, w_0) = ({w}, {w_0})")

    # 二乗損失の計算
    loss = calculate_simple_regression_squared_loss(x, y, w, w_0)
    print("二乗誤差 = {}".format(loss))

    # プロット
    plot_simple_regression(x, y, w, w_0, xlabel='LotArea', ylabel='SalePrice')

if __name__ == '__main__':
    main()
