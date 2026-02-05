import numpy as np
def predict_batch(model, X_new, threshold=0.5):
    probs = model.predict_proba(X_new)[:,1]
    preds = (probs >=threshold).astype(int)
    return probs,preds