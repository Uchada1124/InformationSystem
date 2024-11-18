'''
以下の特徴量のすべてのペアに対して, クロスバリデーションを行うことで最も予測誤差が小さいペアを採択する. 

TotalBsmtSF: Total square feet of basement area

1stFlrSF: First Floor square feet
 
2ndFlrSF: Second floor square feet

GrLivArea: Above grade (ground) living area square feet

GarageArea: Size of garage in square feet
'''

import pandas as pd
import numpy as np

import Python.Regression.multiple_regression as multiple_regression
import Python.Regression.cross_validetion_utils as cross_validetion_utils

def get_feature_pairs(features):
    feature_pairs = []
    n = len(features)
    
    for i in range(1, 2**n):
        if bin(i).count("1") == 2:
            pair = [features[j] for j in range(n) if (i >> j) & 1]
            feature_pairs.append(pair)
    
    return feature_pairs

def main():
    train_df = pd.read_csv('./data/HousePricesAdvancedRegressionTechniques/train.csv')

    features = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea']

    y = train_df['SalePrice'].tolist()
    # 標準化
    y = (np.array(y) - np.mean(y)) / np.std(y)
    
    best_loss = float('inf')
    best_pair = None

    for feature_pair in get_feature_pairs(features):
        x_1 = train_df[feature_pair[0]].tolist()
        x_2 = train_df[feature_pair[1]].tolist()

        # 標準化
        x_1 = (np.array(x_1) - np.mean(x_1)) / np.std(x_1)
        x_2 = (np.array(x_2) - np.mean(x_2)) / np.std(x_2)
        
        # 各ペアに対してクロスバリデーションを実行
        folds = cross_validetion_utils.k_fold_split([x_1, x_2], y, k=5, seed=42)
        
        losses = []
        for i, (x_train, y_train, x_test, y_test) in enumerate(folds):
            x_1_train, x_2_train = x_train
            x_1_test, x_2_test = x_test
            
            # 訓練データを行列形式に変換
            ones_train = np.ones(len(x_1_train))
            X_train = np.column_stack((ones_train, x_1_train, x_2_train))
            
            # テストデータも同様に変換
            ones_test = np.ones(len(x_1_test))
            X_test = np.column_stack((ones_test, x_1_test, x_2_test))
            
            # 重回帰の計算
            w = multiple_regression.calculate_multiple_regression_weights(X_train, y_train)
            loss = multiple_regression.calculate_multiple_regression_squared_loss(X_test, y_test, w)
            losses.append(loss)
        
        avg_loss = np.mean(losses)
        print(f"Feature pair: {feature_pair}, Average Loss: {avg_loss}")

        # 最も良いペアを更新
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_pair = feature_pair

    print(f"\nBest feature pair: {best_pair}, with loss: {best_loss}")

    # 最良の特徴量ペアで重回帰モデルをプロット
    x_1 = train_df[best_pair[0]].tolist()
    x_2 = train_df[best_pair[1]].tolist()

    # 正規化
    x_1 = (np.array(x_1) - np.mean(x_1)) / np.std(x_1)
    x_2 = (np.array(x_2) - np.mean(x_2)) / np.std(x_2)

    ones = np.ones(len(x_1))
    X = np.column_stack((ones, x_1, x_2))

    # 回帰係数を計算し、プロット
    w = multiple_regression.calculate_multiple_regression_weights(X, y)
    multiple_regression.plot_multiple_regression(X, y, w, xlabel=best_pair[0], ylabel=best_pair[1], zlabel='SalePrice')

if __name__ == '__main__':
    main()
