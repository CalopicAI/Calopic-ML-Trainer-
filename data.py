# data.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_tabular(path, target, test_size=0.2, seed=42):
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found in {path}")

    X = df.drop(columns=[target])
    y = df[target]

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    X = X.fillna(0)

    le = None
    if y.dtype == object or str(y.dtype) == "category":
        le = LabelEncoder()
        y = le.fit_transform(y)

    stratify = y if le is not None else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, le