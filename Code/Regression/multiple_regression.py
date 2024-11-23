import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 重回帰を計算する関数
def calculate_multiple_regression_weights(X, y):
    w = np.linalg.solve(X.T@X, X.T@y)
    return w

# 二乗損失を計算する関数
def calculate_multiple_regression_squared_loss(X, y, w):
    # 予測値を計算
    y_pred = X @ w
    
    # 二乗損失を計算
    total_loss = np.sum((y - y_pred) ** 2)
    return total_loss / len(y)

# プロットする関数
def plot_multiple_regression(X, y, w, xlabel="Feature 1 (x1)", ylabel="Feature 2 (x2)", zlabel="Target (y)"):
    x_1 = X[:, 1]  # 第1特徴量
    x_2 = X[:, 2]  # 第2特徴量
    y_pred = X @ w  # 重回帰による予測値
    
    # 3Dプロットの設定
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # データ点をプロット
    ax.scatter(x_1, x_2, y, label=None, color="blue")
    
    # 重回帰平面を描画
    x1_surf, x2_surf = np.meshgrid(np.linspace(min(x_1), max(x_1), 10), np.linspace(min(x_2), max(x_2), 10))
    y_surf = w[0] + w[1] * x1_surf + w[2] * x2_surf
    ax.plot_surface(x1_surf, x2_surf, y_surf, color="red", alpha=0.5, rstride=100, cstride=100)

    # 軸ラベルの設定
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    plt.title("Multiple Linear Regression with Squared Errors (3D)")
    plt.legend()
    plt.show()

def main():
    train_df = pd.read_csv('./data/HousePricesAdvancedRegressionTechniques/train.csv')

    x_1 = train_df['LotArea'][:20].tolist()
    x_2 = train_df['1stFlrSF'][:20].tolist()
    ones = np.ones(len(x_1))
    X = np.column_stack((ones, x_1, x_2))
    y = train_df['SalePrice'][:20].tolist()
    
    # x_1 = train_df['LotArea'].tolist()
    # x_2 = train_df['1stFlrSF'].tolist()
    # ones = np.ones(len(x_1))
    # X = np.column_stack((ones, x_1, x_2))
    # y = train_df['SalePrice'].tolist()

    # 回帰係数の計算
    w = calculate_multiple_regression_weights(X, y)
    print(f"w = ({w}")

    # 二乗損失の計算
    loss = calculate_multiple_regression_squared_loss(X, y, w)
    print("二乗誤差 = {}".format(loss))

    # プロット
    plot_multiple_regression(X, y, w, xlabel='LotArea', ylabel='1stFlrSF', zlabel='SalePrice')

if __name__ == '__main__':
    main()
