import pandas as pd
import numpy as np

import Python.Regression.polynomial_regression as polynomial_regression
import Python.Regression.cross_validetion_utils as cross_validetion_utils
import Python.Regression.ridge_regression as ridge_regression

def main():
    train_df = pd.read_csv('./data/HousePricesAdvancedRegressionTechniques/train.csv')
    
    x = train_df['LotArea'].tolist()
    y = train_df['SalePrice'].tolist()

    # 正規化
    x = (np.array(x) - np.mean(x)) / np.std(x)
    y = (np.array(y) - np.mean(y)) / np.std(y)
    
    # 5分割のクロスバリデーション用データを作成
    folds = cross_validetion_utils.k_fold_split(x, y, k=5, seed=42)
    
    # 正則化パラメータ α の候補
    alphas = [0.01, 0.1, 1, 10, 100]
    degrees = [2, 3, 4, 5, 6, 9]
    
    best_alpha = None
    best_degree = None
    best_loss = float('inf')
    
    for degree in degrees:
        print(f"\nTesting degree = {degree}")
        X_design = np.vstack([np.array(x)**j for j in range(degree + 1)]).T

        for alpha in alphas:
            losses = []
            print(f"  Testing alpha = {alpha}")

            for i, (x_train, y_train, x_test, y_test) in enumerate(folds):
                # デザイン行列を構築
                X_train = np.vstack([np.array(x_train)**j for j in range(degree + 1)]).T
                X_test = np.vstack([np.array(x_test)**j for j in range(degree + 1)]).T
                
                # Ridge回帰の係数を計算
                w = ridge_regression.calculate_ridge_regression_weights(X_train, y_train, alpha)
                
                # テストデータでの二乗損失を計算
                loss = polynomial_regression.calculate_polynomial_regressionsquared_loss(X_test, y_test, w)
                losses.append(loss)
                print(f"    Fold {i + 1}, Loss = {loss}")

            avg_loss = np.mean(losses)
            print(f"  Average Loss for alpha {alpha} and degree {degree}: {avg_loss}")

            # 最適な α と次数を更新
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_alpha = alpha
                best_degree = degree

    print(f"\nBest alpha: {best_alpha}, Best degree: {best_degree}, with loss: {best_loss}")

    # 最適なαと次数でRidge回帰を実行し、プロット
    X = np.vstack([x**i for i in range(best_degree + 1)]).T
    w = ridge_regression.calculate_ridge_regression_weights(X, y, best_alpha)
    polynomial_regression.plot_polynomial_regression(x, y, w, xlabel='LotArea', ylabel='SalePrice')

if __name__ == '__main__':
    main()
