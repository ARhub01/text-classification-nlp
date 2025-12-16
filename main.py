from src.preprocess import load_and_preprocess
from src.models import build_lstm, build_gru
from src.train import train_model
from src.evaluate import evaluate_model
from src.baseline_ml import train_baseline


def main():
    # 1. Preprocess Data
    X_train, X_test, y_train, y_test, vocab_size, max_len = load_and_preprocess('data/raw/imdb_reviews.csv')

    # 2. Train LSTM
    lstm_model = build_lstm(vocab_size, embed_dim=100, max_len=max_len)
    train_model(lstm_model, X_train, y_train, 'results/models/lstm_model.h5')

    # 3. Train GRU
    gru_model = build_gru(vocab_size, embed_dim=100, max_len=max_len)
    train_model(gru_model, X_train, y_train, 'results/models/gru_model.h5')

    # 4. Evaluate
    print("\nLSTM Evaluation:")
    evaluate_model('results/models/lstm_model.h5', X_test, y_test)
    print("\nGRU Evaluation:")
    evaluate_model('results/models/gru_model.h5', X_test, y_test)

    # 5. ML baseline
    print("\nLogistic Regression Baseline:")
    train_baseline(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
