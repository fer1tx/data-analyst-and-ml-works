from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_and_split(seed=42, test_size=0.2):
    df = load_breast_cancer()
    X = df.data
    y = df.target

    X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=test_size,random_state=seed)
    return X_train,X_test,y_train,y_test


def take_new_batch(X_test, y_test, n=100):
    n = min(n, len(y_test))
    return X_test[:n], y_test[:n]