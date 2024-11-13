import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 多項式回帰を計算する関数
def calculate_polynomial_regression_weights(X, y):
    w = np.linalg.solve(X.T @ X, X.T @ y)
    return w

# 二乗損失を計算する関数
def calculate_squared_loss(X, y, w):
    y_pred = X @ w
    total_loss = np.sum((y - y_pred) ** 2) / len(y)
    return total_loss

# プロットする関数
def plot_polynomial_regression(x, y, w, xlabel=None, ylabel=None):
    # データポイントをプロット
    plt.scatter(x, y, label="Data Points")
    
    # 回帰曲線をプロット
    x_vals = np.linspace(min(x), max(x), 100)
    X_vals = np.vstack([x_vals**i for i in range(len(w))]).T
    y_vals = X_vals @ w
    plt.plot(x_vals, y_vals, color="red", label="Regression Curve")
    
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.legend()
    plt.title("Polynomial Regression")
    plt.show()

def main():
    train_df = pd.read_csv('./data/HousePricesAdvancedRegressionTechniques/train.csv')

    # x = train_df['LotArea'][:20].tolist()
    # y = train_df['SalePrice'][:20].tolist()

    x = train_df['LotArea'].tolist()
    y = train_df['SalePrice'].tolist()
    
    # 正規化
    x = (np.array(x) - np.mean(x)) / np.std(x)
    y = (np.array(y) - np.mean(y)) / np.std(y)

    # 回帰係数の計算 (2次多項式を例に)
    degrees = [1, 2, 3, 4, 5, 6, 9]
    for degree in degrees:
        print(f"\n Degree = {degree}")
        X = np.vstack([x**i for i in range(degree + 1)]).T
        w = calculate_polynomial_regression_weights(X, y)
        print(f"回帰係数 w = {w}")

        # 二乗損失の計算
        loss = calculate_squared_loss(X, y, w)
        print("二乗誤差 = {}".format(loss))

        # プロット
        plot_polynomial_regression(x, y, w, xlabel='LotArea', ylabel='SalePrice')

if __name__ == '__main__':
    main()
