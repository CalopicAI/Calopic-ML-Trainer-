# models.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def build_model(name, **kwargs):
    if name == "logreg":
        return LogisticRegression(max_iter=1000, **kwargs)
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=42, **kwargs)
    raise ValueError(f"unknown model: {name}")