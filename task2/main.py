import json
from data import load_and_split, take_new_batch
from train import train_model
from predict import predict_batch
from eval import evaluate_predictions
from visualize import plot_confusion, plot_probs_vs_true

def main():
    X_train, X_test, y_train, y_test = load_and_split(seed=42)
    model = train_model(X_train, y_train)
    X_new, y_true = take_new_batch(X_test, y_test, n=100)

    threshold = 0.5
    probs, preds = predict_batch(model, X_new, threshold=threshold)
    metrics = evaluate_predictions(y_true, probs, preds)

    results = {
        "n_new": int(len(y_true)),
        "threshold": float(threshold),
        **metrics
    }

    with open("new_batch_results.json", "w") as f:
        json.dump(results, f, indent=4)

    plot_confusion(y_true, preds)
    plot_probs_vs_true(probs, y_true, threshold=threshold)

if __name__ == "__main__":
    main()