import pandas as pd
import matplotlib.pyplot as plt

def main():
    train_df = pd.read_csv('../data/HousePricesAdvancedRegressionTechniques/train.csv')
    print(train_df.head())
    print(train_df.isnull().all())

    plt.scatter(train_df['LotArea'][:20], train_df['SalePrice'][:20])
    plt.xlabel('LotArea')
    plt.ylabel('SalePrice')
    plt.show()
    
if __name__ == '__main__':
    main()