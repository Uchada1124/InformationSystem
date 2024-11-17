import pandas as pd
import numpy as np

import Python.Regression.polynomial_regression as polynomial_regression
import Python.Regression.cross_validetion_utils as cross_validetion_utils

def main():
    train_df = pd.read_csv('./data/HousePricesAdvancedRegressionTechniques/train.csv')
    
    x = train_df['LotArea'].tolist()
    y = train_df['SalePrice'].tolist()

    # 正規化
    x = (np.array(x) - np.mean(x)) / np.std(x)
    y = (np.array(y) - np.mean(y)) / np.std(y)
    
    # 5分割のクロスバリデーション用データを作成
    folds = cross_validetion_utils.k_fold_split(x, y, k=5, seed=42)
    
    # 各分割のトレーニングセットとテストセットを表示
    for i, (x_train, y_train, x_test, y_test) in enumerate(folds):
        print(f"\nFold {i + 1}")
        print("Train X:", x_train[:5], "...")
        print("Train Y:", y_train[:5], "...")
        print("Test X:", x_test[:5], "...")
        print("Test Y:", y_test[:5], "...")

    # 次数ごとの二乗誤差を計算
    degrees = [1, 2, 3, 4, 5, 6, 9]
    avg_losses = {}
    for degree in degrees:
        print(f"\nDegree = {degree}")
        losses = []
        
        for i, (x_train, y_train, x_test, y_test) in enumerate(folds):
            # デザイン行列を構築
            X_train = np.vstack([np.array(x_train)**j for j in range(degree + 1)]).T
            X_test = np.vstack([np.array(x_test)**j for j in range(degree + 1)]).T
            
            # 回帰係数を計算
            w = polynomial_regression.calculate_polynomial_regression_weights(X_train, y_train)
            
            # テストデータでの二乗誤差を計算
            loss = polynomial_regression.calculate_polynomial_regressionsquared_loss(X_test, y_test, w)
            losses.append(loss)
            print(f"  Fold {i + 1}, Loss = {loss}")

        avg_loss = np.mean(losses)
        avg_losses[degree] = avg_loss
        print(f"Average Loss for degree {degree}: {avg_loss}")

    # 最適な次数を選ぶ
    best_degree = min(avg_losses, key=avg_losses.get)
    print(f"\nBest degree: {best_degree} with average loss: {avg_losses[best_degree]}")

    # 最適な次数でプロット
    X = np.vstack([x**i for i in range(best_degree + 1)]).T
    w = polynomial_regression.calculate_polynomial_regression_weights(X, y)
    polynomial_regression.plot_polynomial_regression(x, y, w, xlabel='LotArea', ylabel='SalePrice')

if __name__ == '__main__':
    main()
