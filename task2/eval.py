from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
def evaluate_predictions(y_true, probs, preds):
    accuracy = accuracy_score(y_true,preds)

    logloss = log_loss(y_true,probs)

    cm = confusion_matrix(y_true,preds).tolist()

    return {
        "accuracy" : accuracy,
        "log loss" : logloss,
        "confusion matrix" : cm
    }