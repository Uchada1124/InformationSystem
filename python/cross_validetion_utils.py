import numpy as np

# Cross validation 用の分割関数
def k_fold_split(x, y, k=5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    if isinstance(x, (list, np.ndarray)) and not isinstance(x[0], (list, np.ndarray)):
        x = [x]
    
    # データのインデックスをランダムにシャッフル
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    
    # 各分割のサイズを計算
    fold_size = len(y) // k
    folds = []

    for i in range(k):
        # テスト用のインデックスを取得
        test_indices = indices[i * fold_size : (i + 1) * fold_size]
        # トレーニング用のインデックスを取得（テスト以外の部分）
        train_indices = np.setdiff1d(indices, test_indices)

        # トレーニングとテストのデータを分割
        x_train = [[feature[idx] for idx in train_indices] for feature in x]
        x_test = [[feature[idx] for idx in test_indices] for feature in x]
        y_train = [y[idx] for idx in train_indices]
        y_test = [y[idx] for idx in test_indices]

        folds.append((x_train, y_train, x_test, y_test))

    return folds
