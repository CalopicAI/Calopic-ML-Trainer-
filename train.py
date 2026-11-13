# train.py
from models import build_model


def train_and_evaluate(name, X_train, y_train, X_test, y_test, **kwargs):
    model = build_model(name, **kwargs)
    model.fit(X_train, y_train)
    acc = float(model.score(X_test, y_test))
    return model, {"test_acc": acc}