from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output

@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def train_model(
    dataset: Input[Dataset],
    model: Output[Model]
):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    import joblib

    df = pd.read_csv(dataset.path)
    X = df.drop(columns=["target"])
    y = df["target"]

    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y)

    joblib.dump(clf, model.path)