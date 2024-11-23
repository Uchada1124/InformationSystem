import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import polynomial_regression

# Ridge回帰を計算する関数
def calculate_ridge_regression_weights(X, y, alpha):
    # 正則化項を追加した行列を解く
    n = X.shape[1]
    w = np.linalg.solve(X.T @ X + alpha * np.eye(n), X.T @ y)
    return w

def main():
    train_df = pd.read_csv('./data/HousePricesAdvancedRegressionTechniques/train.csv')

    x = train_df['LotArea'].tolist()
    y = train_df['SalePrice'].tolist()
    
    # 標準化
    x = (np.array(x) - np.mean(x)) / np.std(x)
    y = (np.array(y) - np.mean(y)) / np.std(y)

    # Ridge回帰のパラメータ設定
    alpha = 1.0  # 正則化パラメータを調整

    # 回帰係数の計算 (2次多項式を例に)
    degrees = [1, 2, 3, 4, 5, 6, 9]
    for degree in degrees:
        print(f"\n Degree = {degree}")
        X = np.vstack([x**i for i in range(degree + 1)]).T
        w = calculate_ridge_regression_weights(X, y, alpha)
        print(f"回帰係数 w = {w}")

        # 二乗損失の計算
        loss = polynomial_regression.calculate_polynomial_regressionsquared_loss(X, y, w)
        print("二乗誤差 = {}".format(loss))

        # プロット
        polynomial_regression.plot_polynomial_regression(x, y, w, xlabel='LotArea', ylabel='SalePrice')

if __name__ == '__main__':
    main()
