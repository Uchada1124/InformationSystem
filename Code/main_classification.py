import Classification.data_utils as data_utils

def main():
    X, y = data_utils.generate_linear_separable_data()
    print(X, y)

if __name__ == '__main__':
    main()