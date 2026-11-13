# make_data.py
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame
df.columns = [
    c.replace(" (cm)", "").replace(" ", "_") for c in df.columns
]
df.to_csv("iris.csv", index=False)
print("wrote iris.csv")