import pandas as pd
import numpy as np

# Cross validetion 用の分割関数
def k_fold_split(x, y, k=5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # データのインデックスをランダムにシャッフル
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    
    # 各分割のサイズを計算
    fold_size = len(x) // k
    folds = []

    for i in range(k):
        # テスト用のインデックスを取得
        test_indices = indices[i * fold_size : (i + 1) * fold_size]
        # トレーニング用のインデックスを取得（テスト以外の部分）
        train_indices = np.setdiff1d(indices, test_indices)

        # トレーニングとテストのデータを分割
        x_train, x_test = [x[idx] for idx in train_indices], [x[idx] for idx in test_indices]
        y_train, y_test = [y[idx] for idx in train_indices], [y[idx] for idx in test_indices]

        folds.append((x_train, y_train, x_test, y_test))

    return folds

def main():
    train_df = pd.read_csv('./data/HousePricesAdvancedRegressionTechniques/train.csv')
    
    x = train_df['LotArea'].tolist()
    y = train_df['SalePrice'].tolist()

    # 正規化
    x = (np.array(x) - np.mean(x)) / np.std(x)
    y = (np.array(y) - np.mean(y)) / np.std(y)
    
    # 5分割のクロスバリデーション用データを作成
    folds = k_fold_split(x, y, k=5, seed=42)
    
    # 各分割のトレーニングセットとテストセットを表示
    for i, (x_train, y_train, x_test, y_test) in enumerate(folds):
        print(f"\nFold {i + 1}")
        print("Train X:", x_train[:5], "...")
        print("Train Y:", y_train[:5], "...")
        print("Test X:", x_test[:5], "...")
        print("Test Y:", y_test[:5], "...")

    

if __name__ == '__main__':
    main()
