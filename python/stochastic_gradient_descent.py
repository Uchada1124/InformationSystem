import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import simple_regression

def stochastic_gradient_descent(x, y, learning_rate=0.01, epochs=10000):
    w = 0  # 傾きの初期値
    w_0 = 0  # 切片の初期値
    n = len(y)
    
    for _ in range(epochs):
        # ランダムにインデックスを選ぶ
        idx = np.random.randint(n)
        x_i = x[idx]
        y_i = y[idx]
        
        # 予測値と誤差
        y_pred = w * x_i + w_0
        error = y_i - y_pred
        
        # 勾配を計算
        w_grad = -2 * error * x_i
        w_0_grad = -2 * error
        
        # パラメータの更新
        w -= learning_rate * w_grad
        w_0 -= learning_rate * w_0_grad

    return w, w_0

def main():
    train_df = pd.read_csv('./data/HousePricesAdvancedRegressionTechniques/train.csv')

    x = train_df['LotArea'][:20].tolist()
    y = train_df['SalePrice'][:20].tolist()

    # 正規化
    x = (np.array(x) - np.mean(x)) / np.std(x)
    y = (np.array(y) - np.mean(y)) / np.std(y)

    # 勾配降下法で回帰係数の推定
    w, w_0 = stochastic_gradient_descent(x, y)
    print(f"(w, w_0) = ({w}, {w_0})")

    # 二乗損失の計算
    loss = simple_regression.calculate_simple_regression_squared_loss(x, y, w, w_0)
    print("二乗誤差 = {}".format(loss))

    # プロット
    simple_regression.plot_simple_regression(x, y, w, w_0, xlabel='LotArea', ylabel='SalePrice')

if __name__ == '__main__':
    main()
