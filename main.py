from market_data import KalshiFeatureEngineer
from sklearn.linear_model import LogisticRegression

series_ticker = "KXOSCARPIC"  # Oscar for Best Picture ?
market_ticker = "KXOSCARPIC-26-HAM"
start_ts = "2025-09-23"
end_ts = "2026-02-15"

def main():
    dataset_builder = KalshiFeatureEngineer(
        series_ticker=series_ticker,
        market_ticker=market_ticker,
    )

    X, y = dataset_builder.build_features()

    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    print("\nFeature columns:")
    print(list(X.columns))
    print()

    X_train, X_val, y_train, y_val = dataset_builder.split_data(train_size=0.8)

    model = LogisticRegression(max_iter=2000)

    model.fit(X_train, y_train)

    print("Train acc:", model.score(X_train, y_train))
    print("Val acc:", model.score(X_val, y_val))


if __name__ == "__main__":
    main()
